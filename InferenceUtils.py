import logging

import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import (
  ScriptedLoadableModule,
  ScriptedLoadableModuleWidget,
  ScriptedLoadableModuleLogic,
)

from PyTorchUtils import PyTorchUtilsLogic
from TorchIOUtils import TorchIOUtilsLogic
from lib.GeneralUtils import showWaitCursor


MODELS_SMOOTHING = 0.2


class InferenceUtils(ScriptedLoadableModule):
  def __init__(self, parent):
    super().__init__(parent)
    self.parent.title = 'Inference Utils'
    self.parent.categories = []
    self.parent.dependencies = []
    self.parent.contributors = [
      "Fernando Perez-Garcia (University College London and King's College London)",
    ]
    self.parent.helpText = 'This module does things.'
    self.parent.acknowledgementText = (
      'This work was was funded by the Engineering and Physical Sciences'
      ' Research Council (​EPSRC) and supported by the UCL Centre for Doctoral'
      ' Training in Intelligent, Integrated Imaging in Healthcare, the UCL'
      ' Wellcome / EPSRC Centre for Interventional and Surgical Sciences (WEISS),'
      ' and the School of Biomedical Engineering & Imaging Sciences (BMEIS)'
      " of King's College London."
    )


class InferenceUtilsWidget(ScriptedLoadableModuleWidget):
  def setup(self):
    super().setup()
    self.logic = InferenceUtilsLogic()
    self.makeGUI()
    self.onSelectors()
    self._model = None

  @property
  def model(self):
    if self._model is None:
      pu = PyTorchUtilsLogic()
      self._model = pu.getPyTorchHubModel(
        self.REPO_OWNER,
        self.REPO_NAME,
        self.MODEL_NAME,
      )
      self._model.eval()  # set to evaluation mode
      self._model.to(pu.getDevice())  # move to GPU if available
    return self._model

  def makeGUI(self):
    self.makeDataLayout()
    self.makeSettingsLayout()

    self.runPushButton = qt.QPushButton('Run')
    self.runPushButton.clicked.connect(self.onRunButton)
    self.runPushButton.setDisabled(True)
    self.layout.addWidget(self.runPushButton)

    self.layout.addStretch(1)

  def makeDataLayout(self):
    self.dataCollapsibleButton = ctk.ctkCollapsibleButton()
    self.dataCollapsibleButton.text = "Data"
    self.layout.addWidget(self.dataCollapsibleButton)
    self.dataLayout = qt.QFormLayout(self.dataCollapsibleButton)

    self.downloadOasisPushButton = qt.QPushButton('Download sample data')
    self.downloadOasisPushButton.clicked.connect(self.onDownloadSampleData)
    self.dataLayout.addWidget(self.downloadOasisPushButton)

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
    self.dataLayout.addRow("Input volume: ", self.inputNodeSelector)

    self.outputNodeSelector = slicer.qMRMLNodeComboBox()
    self.outputNodeSelector.nodeTypes = ['vtkMRMLSegmentationNode']
    self.outputNodeSelector.addEnabled = True
    self.outputNodeSelector.removeEnabled = False
    self.outputNodeSelector.noneEnabled = False
    self.outputNodeSelector.showHidden = False
    self.outputNodeSelector.showChildNodeTypes = False
    self.outputNodeSelector.setMRMLScene(slicer.mrmlScene)
    self.outputNodeSelector.setToolTip('')  # TODO
    self.outputNodeSelector.currentNodeChanged.connect(self.onSelectors)
    self.dataLayout.addRow("Output segmentation: ", self.outputNodeSelector)

  def makeSettingsLayout(self):
    self.settingsCollapsibleButton = ctk.ctkCollapsibleButton()
    self.settingsCollapsibleButton.text = 'Settings'
    self.layout.addWidget(self.settingsCollapsibleButton)
    self.settingsLayout = qt.QFormLayout(self.settingsCollapsibleButton)

    self.mixedPrecisionCheckBox = qt.QCheckBox()
    self.settingsLayout.addRow('Use mixed precision: ', self.mixedPrecisionCheckBox)
    self.mixedPrecisionCheckBox.setChecked(True)

    self.inferenceModeGroupBox = qt.QGroupBox()
    self.settingsLayout.addRow('Inference mode: ', self.inferenceModeGroupBox)
    self.inferenceModeLayout = qt.QHBoxLayout(self.inferenceModeGroupBox)

    self.fullVolumeRadioButton = qt.QRadioButton('Full volume')
    self.inferenceModeLayout.addWidget(self.fullVolumeRadioButton)
    self.patchesRadioButton = qt.QRadioButton('Patches')
    self.inferenceModeLayout.addWidget(self.patchesRadioButton)
    # self.autoRadioButton = qt.QRadioButton('Auto')
    # self.inferenceModeLayout.addWidget(self.autoRadioButton)

    # self.autoRadioButton.setChecked(True)
    self.fullVolumeRadioButton.setChecked(True)

    self.patchesRadioButton.toggled.connect(self.onPatchesButton)

    self.patchesSettingsGroupBox = qt.QGroupBox('Patches settings')
    self.settingsLayout.addRow(self.patchesSettingsGroupBox)
    self.patchesSettingsLayout = qt.QFormLayout(self.patchesSettingsGroupBox)

    self.patchSizeSpinBox = qt.QSpinBox()
    self.patchSizeSpinBox.minimum = 10
    self.patchSizeSpinBox.maximum = 1000
    self.patchSizeSpinBox.value = 128
    self.patchesSettingsLayout.addRow('Patch size: ', self.patchSizeSpinBox)

    self.patchOverlapSpinBox = qt.QSpinBox()
    self.patchOverlapSpinBox.minimum = 0
    self.patchOverlapSpinBox.maximum = 1000
    self.patchOverlapSpinBox.value = 4
    self.patchesSettingsLayout.addRow('Overlap: ', self.patchOverlapSpinBox)

    self.batchSizeSpinBox = qt.QSpinBox()
    self.batchSizeSpinBox.minimum = 1
    self.batchSizeSpinBox.maximum = 100
    self.batchSizeSpinBox.value = 1
    self.patchesSettingsLayout.addRow('Batch size: ', self.batchSizeSpinBox)

    self.onPatchesButton()

  def getDataNodes(self):
    inputNode = self.inputNodeSelector.currentNode()
    outputNode = self.outputNodeSelector.currentNode()
    return inputNode, outputNode

  def onSelectors(self):
    inputNode, outputNode = self.getDataNodes()
    enable = inputNode is not None and outputNode is not None
    self.runPushButton.setEnabled(enable)

  def onPatchesButton(self):
    self.patchesSettingsGroupBox.setVisible(self.patchesRadioButton.isChecked())

  def onDownloadSampleData(self):
    # TODO: download MRHead
    return

  def onRunButton(self):
    if not self.logic.confirmDeviceOk():
      return
    inputVolumeNode, outputSegmentationNode = self.getDataNodes()

    try:
      with showWaitCursor():
        self.logic.parcellate(
          self.model,
          inputVolumeNode,
          outputSegmentationNode,
          patchBased=self.patchesRadioButton.isChecked(),
          useMixedPrecision=self.mixedPrecisionCheckBox.isChecked(),
          patchSize=self.patchSizeSpinBox.value,
          patchOverlap=self.patchOverlapSpinBox.value,
          batchSize=self.batchSizeSpinBox.value,
        )
        self.logic.hideBlackSegments(outputSegmentationNode)
    except Exception as e:
      slicer.util.errorDisplay(f'Error running segmentation:\n{e}')


class InferenceUtilsLogic(ScriptedLoadableModuleLogic):
  def __init__(self):
    self.torchLogic = PyTorchUtilsLogic()
    self.torchioLogic = TorchIOUtilsLogic()

  def confirmDeviceOk(self):
    if self.torchLogic.getDevice() == 'cpu':
      text = (
        'Inference might take some minutes as it will run on the CPU.'
        ' Do you want to continue anyway?'
      )
      run = slicer.util.confirmYesNoDisplay(text)
    else:
      run = True
    return run

  def inferVolume(self, model, image):
    tio = self.torchioLogic.torchio
    inputTensor = image.data.unsqueeze(0)  # add batch dimension
    logging.info('Sending tensor to device...')
    inputTensor = inputTensor.to(self.torchLogic.getDevice())
    logits = model(inputTensor)
    labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True).byte().cpu()
    output = tio.LabelMap(tensor=labels[0], affine=image.affine)
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
    device = self.torchLogic.getDevice()
    imageName = 'mri'
    subject = tio.Subject({imageName: inputImage})
    gridSampler = tio.inference.GridSampler(subject, patchSize, patchOverlap)
    patchLoader = torch.utils.data.DataLoader(gridSampler, batch_size=batchSize)
    aggregator = tio.inference.GridAggregator(gridSampler)
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
        progressDialog.setLabelText(f'Processing patch {i+1}/{numBatches}...')
        slicer.app.processEvents()  # necessary?
      inputTensor = patchesBatch[imageName][tio.DATA].to(device)
      locations = patchesBatch[tio.LOCATION]
      logits = model(inputTensor)
      labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True).cpu()
      aggregator.add_batch(labels, locations)

    if showProgress:
      progressDialog.setValue(numBatches)
      slicer.app.processEvents()  # necessary?
      progressDialog.close()

    outputTensor = aggregator.get_output_tensor()
    image = tio.LabelMap(tensor=outputTensor, affine=inputImage.affine)
    return image

  @staticmethod
  def labelMapToSegmentation(labelMapNode, segmentationNode):
    segmentationsLogic = slicer.modules.segmentations.logic()
    segmentationsLogic.ImportLabelmapToSegmentationNode(labelMapNode, segmentationNode)

    rule = slicer.vtkBinaryLabelmapToClosedSurfaceConversionRule
    smoothingParameter = rule.GetSmoothingFactorParameterName()
    segmentation = segmentationNode.GetSegmentation()
    segmentation.SetConversionParameter(smoothingParameter, str(MODELS_SMOOTHING))
    segmentationNode.CreateClosedSurfaceRepresentation()

  @staticmethod
  def getSegmentIDs(segmentationNode):
    segmentation = segmentationNode.GetSegmentation()
    array = vtk.vtkStringArray()
    segmentation.GetSegmentIDs(array)
    ids = [array.GetValue(i) for i in range(array.GetNumberOfValues())]
    return ids
