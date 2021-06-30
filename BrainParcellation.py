import logging
from pathlib import Path

import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from InferenceUtils import InferenceUtilsWidget, InferenceUtilsLogic


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


class BrainParcellationWidget(InferenceUtilsWidget):
  REPO_OWNER = 'fepegar'
  REPO_NAME = 'highresnet'
  MODEL_NAME = 'highres3dnet'

  def setup(self):
    super().setup()
    self.logic = BrainParcellationLogic()

  def onDownloadSampleData(self):
    imagePath = self.logic.downloadOasis()
    slicer.util.loadVolume(str(imagePath))


class BrainParcellationLogic(InferenceUtilsLogic):
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
    transforms = (
      tio.ToCanonical(),  # to RAS
      tio.Resample(image_interpolation=interpolation),  # to 1 mm iso
      tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    )
    transform = tio.Compose(transforms)
    subject = tio.Subject({key: torchIOImage})  # for the histogram standardization
    transformed = transform(subject)[key]
    return transformed

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
