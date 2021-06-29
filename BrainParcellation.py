import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
  ScriptedLoadableModule,
  ScriptedLoadableModuleWidget,
  ScriptedLoadableModuleLogic,
  ScriptedLoadableModuleTest,
)

from PyTorchUtils import PyTorchUtilsLogic
from TorchIOUtils import TorchIOUtilsLogic


REPO_OWNER = 'fepegar'
REPO_NAME = 'highresnet'
MODEL_NAME = 'highres3dnet'


class BrainParcellation(ScriptedLoadableModule):
  def __init__(self, parent):
    super().__init__(parent)
    self.parent.title = "Brain Parcellation"
    self.parent.categories = ["Segmentation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Fernando Perez-Garcia (University College London)"]
    self.parent.helpText = 'This module does this and that.'
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = (
      'This work was was funded by the Engineering and Physical Sciences'
      ' Research Council (​EPSRC)'
    )


class BrainParcellationWidget(ScriptedLoadableModuleWidget):
  def setup(self):
    super().setup()
    self.logic = BrainParcellationLogic()
    self.makeGUI()
    self.onSelectors()
    self._model = None

  @property
  def model(self):
    if self._model is None:
      pu = PyTorchUtilsLogic()
      self._model = pu.getPyTorchHubModel(REPO_OWNER, REPO_NAME, MODEL_NAME)
      self._model.to(pu.getDevice())
    return self._model

  def makeGUI(self):
    self.inputCollapsibleButton = ctk.ctkCollapsibleButton()
    self.inputCollapsibleButton.text = "Input"
    self.layout.addWidget(self.inputCollapsibleButton)
    nodesFormLayout = qt.QFormLayout(self.inputCollapsibleButton)

    self.inputNodeSelector = slicer.qMRMLNodeComboBox()
    self.inputNodeSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.inputNodeSelector.addEnabled = False
    self.inputNodeSelector.removeEnabled = False
    self.inputNodeSelector.noneEnabled = False
    self.inputNodeSelector.showHidden = False
    self.inputNodeSelector.showChildNodeTypes = False
    self.inputNodeSelector.setMRMLScene(slicer.mrmlScene)
    self.inputNodeSelector.setToolTip('')  # TODO
    self.inputNodeSelector.currentNodeChanged.connect(self.onSelectors)
    nodesFormLayout.addRow("Input volume: ", self.inputNodeSelector)

    self.outputNodeSelector = slicer.qMRMLNodeComboBox()
    self.outputNodeSelector.nodeTypes = ['vtkMRMLLabelMapVolumeNode']
    self.outputNodeSelector.addEnabled = True
    self.outputNodeSelector.removeEnabled = False
    self.outputNodeSelector.noneEnabled = False
    self.outputNodeSelector.showHidden = False
    self.outputNodeSelector.showChildNodeTypes = False
    self.outputNodeSelector.setMRMLScene(slicer.mrmlScene)
    self.outputNodeSelector.setToolTip('')  # TODO
    self.outputNodeSelector.currentNodeChanged.connect(self.onSelectors)
    nodesFormLayout.addRow("Output volume: ", self.outputNodeSelector)

    self.runPushButton = qt.QPushButton('Run')
    self.runPushButton.clicked.connect(self.onRunButton)
    self.runPushButton.setDisabled(True)
    nodesFormLayout.addWidget(self.runPushButton)

    self.layout.addStretch(1)

  def getNodes(self):
    inputNode = self.inputNodeSelector.currentNode()
    outputNode = self.outputNodeSelector.currentNode()
    return inputNode, outputNode

  def onSelectors(self):
    inputNode, outputNode = self.getNodes()
    enable = inputNode is not None and outputNode is not None
    self.runPushButton.setEnabled(enable)

  def onRunButton(self):
    inputNode, outputNode = self.getNodes()
    model = None  # TODO
    self.logic.parcellate(self.model, inputNode, outputNode)
    slicer.util.setSliceViewerLayers(label=outputNode.GetID())


class BrainParcellationLogic(ScriptedLoadableModuleLogic):
  def __init__(self):
    self.torchLogic = PyTorchUtilsLogic()
    self.torchioLogic = TorchIOUtilsLogic()

  def parcellate(
      self,
      model,
      inputVolumeNode,
      outputLabelMapNode,  # outputSegmentationNode,  # TODO
      useMixedPrecision=True,
      ):
    torch = self.torchLogic.torch
    inputTorchIOImage = self.torchioLogic.getTorchIOImageFromVolumeNode(inputVolumeNode)
    with torch.no_grad():
      with torch.cuda.amp.autocast(enabled=useMixedPrecision):
        # outputTorchIOImage = self.inferVolume(
        #   model,
        #   inputTorchIOImage,
        #   useMixedPrecision=useMixedPrecision,
        # )
        outputTorchIOImage = self.inferPatches(
          model,
          inputTorchIOImage,
          patchSize=128,
          patchOverlap=0,
          batchSize=1,
        )
    outputLabelMapNode = self.torchioLogic.getVolumeNodeFromTorchIOImage(
      outputTorchIOImage,
      outputLabelMapNode,
    )
    self.setGIFColors(outputLabelMapNode)
    return outputLabelMapNode

  def inferVolume(self, model, torchIOImage):
    # TODO
    tio = self.torchioLogic.torchio
    toFloat = tio.Lambda(lambda x: x.float())
    threshold = tio.Lambda(lambda x: x > x.mean())
    transform = tio.Compose([toFloat, threshold])
    output = transform(torchIOImage)
    return output

  def inferPatches(
      self,
      model,
      inputImage,
      patchSize,
      patchOverlap,
      batchSize,
      showProgress=True,
      ):
    torch = self.torchLogic.torch
    tio = self.torchioLogic.torchio
    imageName = 'mri'
    subject = tio.Subject({imageName: inputImage})
    gridSampler = tio.inference.GridSampler(subject, patchSize, patchOverlap)
    patchLoader = torch.utils.data.DataLoader(gridSampler, batch_size=batchSize)
    aggregator = tio.inference.GridAggregator(gridSampler)
    # TODO: if there is patch overlap, are the labels being (wrongly) averaged?
    if showProgress:
      numBatches = len(patchLoader)
      progressDialog = slicer.util.createProgressDialog(
        value=0,
        maximum=numBatches,
        windowTitle='Running inference...',
      )
    for i, patchesBatch in enumerate(patchLoader):
      if showProgress:
        progressDialog.setValue(i)
        slicer.app.processEvents()  # necessary?
      inputTensor = patchesBatch[imageName][tio.DATA]  # TODO .to(model.device)
      locations = patchesBatch[tio.LOCATION]
      logits = model(inputTensor)
      labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
      # labels = (torch.rand(*tuple(inputTensor.shape), device=inputTensor.device) > 0.5).byte()
      aggregator.add_batch(labels, locations)

    if showProgress:
      progressDialog.setValue(numBatches)
      slicer.app.processEvents()  # necessary?
      progressDialog.close()

    outputTensor = aggregator.get_output_tensor()
    image = tio.LabelMap(tensor=outputTensor, affine=inputImage.affine)
    return image

  def setGIFColors(self, labelMapVolumeNode):
    pass  # TODO
