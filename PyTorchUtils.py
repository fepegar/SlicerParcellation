import logging

import slicer
from slicer.ScriptedLoadableModule import (
  ScriptedLoadableModule,
  ScriptedLoadableModuleLogic,
)

from lib.GeneralUtils import showWaitCursor, peakPythonConsole


class PyTorchUtils(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "PyTorch Utils"
    self.parent.categories = ["Utilities"]
    self.parent.dependencies = []
    self.parent.contributors = ["Fernando Perez-Garcia (University College London)"]
    self.parent.helpText = 'This module does this and that.'
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = (
      'This work was was funded by the Engineering and Physical Sciences'
      ' Research Council (​EPSRC)'
    )


class PyTorchUtilsLogic(ScriptedLoadableModuleLogic):
  def __init__(self):
    self._torch = None

  @property
  def cuda(self):
    return self.getDevice() != 'cpu'

  @property
  def torch(self):
    if self._torch is None:
      logging.info('Importing torch...')
      self._torch = self.importTorch()
    return self._torch

  def importTorch(self):
    try:
      import torch
    except ModuleNotFoundError:
      with showWaitCursor(), peakPythonConsole():
        torch = self.installTorch()
    logging.info(f'PyTorch {torch.__version__} imported correctly')
    logging.info(f'CUDA available: {torch.cuda.is_available()}')
    return torch

  def installTorch(self):
    wheelUrl = self.getTorchUrl()
    slicer.util.pip_install(wheelUrl)
    import torch
    logging.info(f'PyTorch {torch.__version__} installed correctly')
    return torch

  @staticmethod
  def getTorchUrl():
    slicer.util.pip_install('light-the-torch')
    import light_the_torch as ltt
    wheelUrl = ltt.find_links(['torch'])[0]
    return wheelUrl

  def getPyTorchHubModel(self, repoOwner, repoName, modelName, *args, **kwargs):
    # This will fail if dependencies in the corresponding hub.py are not installed
    repo = f'{repoOwner}/{repoName}'
    model = self.torch.hub.load(repo, modelName, *args, pretrained=True, **kwargs)
    return model

  def getDevice(self):
    torch = self.torch
    return torch.device('cuda') if torch.cuda.is_available() else 'cpu'
