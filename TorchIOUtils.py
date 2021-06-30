import logging

import SimpleITK as sitk

import slicer
from slicer.ScriptedLoadableModule import (
  ScriptedLoadableModule,
  ScriptedLoadableModuleLogic,
)
import sitkUtils as su

from PyTorchUtils import PyTorchUtilsLogic
from lib.GeneralUtils import showWaitCursor, peakPythonConsole


class TorchIOUtils(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "TorchIO Utils"
    self.parent.categories = []
    self.parent.dependencies = []
    self.parent.contributors = ["Fernando Perez-Garcia (University College London)"]
    self.parent.helpText = 'This module does this and that.'
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = (
      'This work was was funded by the Engineering and Physical Sciences'
      ' Research Council (​EPSRC)'
    )


class TorchIOUtilsLogic(ScriptedLoadableModuleLogic):
  def __init__(self):
    self._torchio = None
    self.torchLogic = PyTorchUtilsLogic()

  @property
  def torchio(self):
    if self._torchio is None:
      logging.info('Importing torchio...')
      self._torchio = self.importTorchIO()
    return self._torchio

  def importTorchIO(self):
    self.torchLogic.torch  # make sure torch is installed
    try:
      import torchio
    except ModuleNotFoundError:
      with showWaitCursor(), peakPythonConsole():
        torchio = self.installTorchIO()
    logging.info(f'TorchIO {torchio.__version__} imported correctly')
    return torchio

  @staticmethod
  def installTorchIO():
    slicer.util.pip_install('torchio')
    import torchio
    logging.info(f'TorchIO {torchio.__version__} installed correctly')
    return torchio

  def getTorchIOImageFromVolumeNode(self, volumeNode):
    image = su.PullVolumeFromSlicer(volumeNode)
    tio = self.torchio
    if volumeNode.IsA('vtkMRMLScalarVolumeNode'):
      image = sitk.Cast(image, sitk.sitkFloat32)
      class_ = tio.ScalarImage
    elif volumeNode.IsA('vtkMRMLLabelMapVolumeNode'):
      class_ = tio.LabelMap
    tensor, affine = tio.io.sitk_to_nib(image)
    return class_(tensor=tensor, affine=affine)

  @staticmethod
  def getVolumeNodeFromTorchIOImage(image, outputVolumeNode):
    su.PushVolumeToSlicer(image.as_sitk(), targetNode=outputVolumeNode)
    return outputVolumeNode
