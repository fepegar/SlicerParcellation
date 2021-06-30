import logging
import tempfile
from pathlib import Path

import numpy as np

import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from lib.GeneralUtils import showWaitCursor, peakPythonConsole
from InferenceUtils import InferenceUtilsWidget, InferenceUtilsLogic

BLACK = 0, 0, 0
LI_PAPER = 'https://link.springer.com/chapter/10.1007/978-3-319-59050-9_28'
CARDOSO_PAPER = 'https://pubmed.ncbi.nlm.nih.gov/25879909/'
PEREZGARCIA_REPO = 'https://github.com/fepegar/highresnet'


class BrainResectionCavitySegmentation(ScriptedLoadableModule):
  def __init__(self, parent):
    super().__init__(parent)
    self.parent.title = 'Brain Resection Cavity Segmentation'
    self.parent.categories = ['Segmentation']
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


class BrainResectionCavitySegmentationWidget(InferenceUtilsWidget):
  REPO_OWNER = 'fepegar'
  REPO_NAME = 'resseg'
  MODEL_NAME = 'ressegnet'

  def setup(self):
    super().setup()
    self.logic = BrainResectionCavitySegmentationLogic()
    self.colorNode = None

  @staticmethod
  def checkPatchSize(value):
    if value % 8:
      slicer.util.errorDisplay('For this model, the patch size must be disivible by 8')
      return False
    return True

  def onDownloadSampleData(self):
    imagePath = self.logic.downloadBite()
    slicer.util.loadVolume(str(imagePath))

  def onRunButton(self):
    if not self.checkPatchSize(self.patchSizeSpinBox.value):
      return
    self.logic.unet  # make sure the unet package is installed
    super().onRunButton()


class BrainResectionCavitySegmentationLogic(InferenceUtilsLogic):
  def __init__(self):
    super().__init__()
    self._unet = None

  @property
  def unet(self):
    if self._unet is None:
      self.torchLogic.torch  # make sure PyTorch is installed
      self.torchioLogic.torchio  # make sure TorchIO is installed
      self._unet = self.importUnet()
    return self._unet

  def importUnet(self):
    try:
      import unet
    except ModuleNotFoundError:
      with showWaitCursor(), peakPythonConsole():
        unet = self.installUnet()
    logging.info(f'UNet {unet.__version__} imported correctly')
    return unet

  @staticmethod
  def installUnet():
    slicer.util.pip_install('unet==0.7.7')  # from https://github.com/fepegar/resseg#trained-model
    import unet
    logging.info(f'UNet {unet.__version__} installed correctly')
    return unet

  def preprocess(self, torchIOImage, interpolation='linear'):
    tio = self.torchioLogic.torchio
    image_name = 't1'  # for example
    subject = tio.Subject({image_name: torchIOImage})
    landmarks = np.array([
        0.        ,   0.31331614,   0.61505419,   0.76732501,
        0.98887953,   1.71169384,   3.21741126,  13.06931455,
        32.70817796,  40.87807389,  47.83508873,  63.4408591 ,
      100.
    ])
    transforms = (
        tio.ToCanonical(),
        tio.HistogramStandardization({image_name: landmarks}),
        tio.Resample(image_interpolation=interpolation),  # to 1 mm iso
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.EnsureShapeMultiple(8, method='crop'),  # for the UNet
    )
    transform = tio.Compose(transforms)
    subject = tio.Subject({image_name: torchIOImage})  # for the histogram standardization
    transformed = transform(subject)[image_name]
    return transformed

  def postprocess(self, outputImage, inputImage):
    tio = self.torchioLogic.torchio
    largestComponent = tio.KeepLargestComponent()(outputImage)
    outputInInputSpace = tio.Resample(inputImage)(largestComponent)
    return outputInInputSpace

  @staticmethod
  def downloadBite():
    url = 'https://github.com/fepegar/resseg/raw/master/sample_data/04_postop_mri_bite_u8.nii.gz'
    tempDir = Path(tempfile.gettempdir())
    filePath = tempDir / '04_postop_mri.nii.gz'
    slicer.util.downloadFile(url, str(filePath))
    return filePath
