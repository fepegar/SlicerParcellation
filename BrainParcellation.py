import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
  ScriptedLoadableModule,
  ScriptedLoadableModuleWidget,
  ScriptedLoadableModuleLogic,
  ScriptedLoadableModuleTest,
)


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
    self.logic.parcellate(model, inputNode, outputNode)
    slicer.util.setSliceViewerLayers(label=outputNode.GetID())


class BrainParcellationLogic(ScriptedLoadableModuleLogic):
  def parcellate(
      self,
      model,
      inputVolumeNode,
      outputLabelMapNode,  # outputSegmentationNode,  # TODO
      useMixedPrecision=True,
      ):
    from TorchIOUtils import TorchIOUtilsLogic
    tioLogic = TorchIOUtilsLogic()
    inputTorchIOImage = tioLogic.getTorchIOImageFromVolumeNode(inputVolumeNode)
    outputTorchIOImage = self.infer(
      model,
      inputTorchIOImage,
      useMixedPrecision=useMixedPrecision,
    )
    outputLabelMapNode = tioLogic.getVolumeNodeFromTorchIOImage(
      outputTorchIOImage,
      outputLabelMapNode,
    )
    self.setGIFColors(outputLabelMapNode)
    return outputLabelMapNode

  def infer(self, model, torchIOImage, useMixedPrecision):
    # TODO
    from TorchIOUtils import TorchIOUtilsLogic
    tio = TorchIOUtilsLogic().torchio
    toFloat = tio.Lambda(lambda x: x.float())
    threshold = tio.Lambda(lambda x: x > x.mean())
    transform = tio.Compose([toFloat, threshold])
    output = transform(torchIOImage)
    return output

  def setGIFColors(self, labelMapVolumeNode):
    pass  # TODO
