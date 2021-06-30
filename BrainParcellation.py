import logging
from os import stat
from pathlib import Path

import numpy as np

import qt
import ctk
import vtk
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
LI_LANDMARKS = (0.0, 8.1, 15.5, 18.7, 21.5, 26.1, 30.0, 33.8, 38.2, 40.7, 44.0, 58.4, 100.0)
SMOOTHING = 0.2
BLACK = 0, 0, 0
LI_PAPER = 'https://link.springer.com/chapter/10.1007/978-3-319-59050-9_28'
CARDOSO_PAPER = 'https://pubmed.ncbi.nlm.nih.gov/25879909/'
PEREZGARCIA_REPO = 'https://github.com/fepegar/highresnet'


class BrainParcellation(ScriptedLoadableModule):
  def __init__(self, parent):
    super().__init__(parent)
    self.parent.title = 'Brain Parcellation'
    self.parent.categories = ['Segmentation']
    self.parent.dependencies = []
    self.parent.contributors = [
      "Fernando Perez-Garcia (University College London and King's College London)",
    ]
    self.parent.helpText = (
      'Brain parcellation using deep learning.'
      f'<p>Paper: <a href="{LI_PAPER}">Li et al. 2017, On the Compactness,'
      ' Efficiency, and Representation of 3D Convolutional Networks: Brain'
      ' Parcellation as a Pretext Task</a>.</p>'
      f'<p>GIF parcellation: <a href="{CARDOSO_PAPER}">Cardoso et al. 2015,'
      ' Geodesic Information Flows: Spatially-Variant Graphs and Their'
      ' Application to Segmentation and Fusion</a>.</p>'
      f'<p>PyTorch implementation: <a href="{PEREZGARCIA_REPO}">Perez-Garcia 2019,'
      ' highresnet GitHub repository</a>.</p>'
    )
    self.parent.acknowledgementText = (
      'This work was was funded by the Engineering and Physical Sciences'
      ' Research Council (​EPSRC) and supported by the UCL Centre for Doctoral'
      ' Training in Intelligent, Integrated Imaging in Healthcare, the UCL'
      ' Wellcome / EPSRC Centre for Interventional and Surgical Sciences (WEISS),'
      ' and the School of Biomedical Engineering & Imaging Sciences (BMEIS)'
      " of King's College London."
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
    self.downloadOasisPushButton.clicked.connect(self.onDownloadOasis)
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

  def onDownloadOasis(self):
    imagePath = self.logic.downloadOasis()
    slicer.util.loadVolume(str(imagePath))

  def onRunButton(self):
    if not self.logic.confirmDeviceOk():
      return
    inputVolumeNode, outputSegmentationNode = self.getDataNodes()
    try:
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
      slicer.util.errorDisplay(f'Error running parcellation:\n{e}')

class BrainParcellationLogic(ScriptedLoadableModuleLogic):
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

  def parcellate(
      self,
      model,
      inputVolumeNode,
      outputSegmentationNode,
      patchBased=True,
      useMixedPrecision=True,
      patchSize=None,
      patchOverlap=None,
      batchSize=None,
      ):
    tio = self.torchioLogic.torchio
    torch = self.torchLogic.torch
    torch.set_grad_enabled(False)

    logging.info('Creating TorchIO image...')
    inputImage = self.torchioLogic.getTorchIOImageFromVolumeNode(inputVolumeNode)

    logging.info('Preprocessing input...')
    preprocessedImage = self.preprocess(inputImage)

    logging.info('Running inference...')
    with torch.cuda.amp.autocast(enabled=useMixedPrecision):
      if patchBased:
        outputTorchIOImage = self.inferPatches(
          model,
          preprocessedImage,
          patchSize=patchSize,
          patchOverlap=patchOverlap,
          batchSize=batchSize,
        )
      else:
        outputTorchIOImage = self.inferVolume(
          model,
          preprocessedImage,
        )

    logging.info('Postprocessing output...')
    outputInInputSpace = tio.Resample(inputImage)(outputTorchIOImage)

    logging.info('Creating label map...')
    labelMapNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    labelMapNode = self.torchioLogic.getVolumeNodeFromTorchIOImage(outputInInputSpace, labelMapNode)
    labelMapNode.CreateDefaultDisplayNodes()
    self.setGIFColors(labelMapNode)

    logging.info('Creating segmentation and meshes...')
    self.labelMapToSegmentation(labelMapNode, outputSegmentationNode)
    slicer.mrmlScene.RemoveNode(labelMapNode)
    return outputSegmentationNode

  def preprocess(self, torchIOImage, interpolation='linear'):
    key = 't1'
    tio = self.torchioLogic.torchio
    landmarks = {key: np.array(LI_LANDMARKS)}
    transforms = (
      tio.ToCanonical(),  # to RAS
      tio.Resample(image_interpolation=interpolation),  # to 1 mm iso
      tio.HistogramStandardization(landmarks=landmarks),
      tio.ZNormalization(),
    )
    transform = tio.Compose(transforms)
    subject = tio.Subject({key: torchIOImage})  # for the histogram standardization
    transformed = transform(subject)[key]
    return transformed

  def inferVolume(self, model, image):
    tio = self.torchioLogic.torchio
    inputTensor = image.data.unsqueeze(0)  # add batch dimension
    logging.info('Sending tensor to device...')
    inputTensor = inputTensor.to(self.torchLogic.getDevice())
    logits = model(inputTensor)
    labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True).cpu()
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
    # TODO: if there is patch overlap, are the labels being (incorrectly) averaged?
    if showProgress:
      numBatches = len(patchLoader)
      progressDialog = slicer.util.createProgressDialog(
        value=0,
        maximum=numBatches,
        windowTitle='Running inference...',
      )
    for i, patchesBatch in enumerate(patchLoader):
      if i != numBatches // 2: continue  # for debugging
      if showProgress:
        progressDialog.setValue(i)
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

  def getColorNode(self):
    colorTablePath = Path(__file__).parent / 'GIFNiftyNet.ctbl'
    try:
      colorNode = slicer.util.getNode(colorTablePath.stem)
    except slicer.util.MRMLNodeNotFoundException:
      colorNode = slicer.util.loadColorTable(str(colorTablePath))
    return colorNode

  def setGIFColors(self, labelMapVolumeNode):
    colorNode = self.getColorNode()
    displayNode = labelMapVolumeNode.GetDisplayNode()
    displayNode.SetAndObserveColorNodeID(colorNode.GetID())

  @staticmethod
  def labelMapToSegmentation(labelMapNode, segmentationNode):
    segmentationsLogic = slicer.modules.segmentations.logic()
    segmentationsLogic.ImportLabelmapToSegmentationNode(labelMapNode, segmentationNode)

    rule = slicer.vtkBinaryLabelmapToClosedSurfaceConversionRule
    smoothingParameter = rule.GetSmoothingFactorParameterName()
    segmentation = segmentationNode.GetSegmentation()
    segmentation.SetConversionParameter(smoothingParameter, str(SMOOTHING))
    segmentationNode.CreateClosedSurfaceRepresentation()

  @staticmethod
  def getSegmentIDs(segmentationNode):
    segmentation = segmentationNode.GetSegmentation()
    array = vtk.vtkStringArray()
    segmentation.GetSegmentIDs(array)
    ids = [array.GetValue(i) for i in range(array.GetNumberOfValues())]
    return ids

  def hideBlackSegments(self, segmentationNode):
    segmentation = segmentationNode.GetSegmentation()
    displayNode = segmentationNode.GetDisplayNode()
    for segmentID in self.getSegmentIDs(segmentationNode):
      segment = segmentation.GetSegment(segmentID)
      if segment.GetColor() == BLACK:
        logging.info(f'Hiding {segmentID}...')
        displayNode.SetSegmentVisibility(segmentID, False)

  @staticmethod
  def downloadOasis():
    """
    http://blog.ppkt.eu/2014/06/python-urllib-and-tarfile/
    """
    import os
    import urllib
    import shutil
    import tarfile
    import tempfile
    fileName = 'OAS1_0145_MR2_mpr_n4_anon_sbj_111.nii.gz'
    dst = Path(tempfile.gettempdir()) / fileName
    if not dst.is_file():
      logging.info('Downloading OASIS image...')
      url = 'https://github.com/NifTK/NiftyNetModelZoo/raw/5-reorganising-with-lfs/highres3dnet_brain_parcellation/data.tar.gz'
      fileTmp = urllib.request.urlretrieve(url, filename=None)[0]
      tar = tarfile.open(fileTmp)
      tempDir = Path(slicer.util.tempDirectory())
      tar.extractall(tempDir)
      src = tempDir / fileName
      os.rename(src, dst)
      shutil.rmtree(tempDir)
    return dst
