from slicer.ScriptedLoadableModule import (
  ScriptedLoadableModule,
  ScriptedLoadableModuleWidget,
  ScriptedLoadableModuleLogic,
  ScriptedLoadableModuleTest,
)

from PyTorchUtils import PyTorchUtilsLogic


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
    outputLabelMapNode = tioLogic.getVolumeNodeFromTorchIOImage(outputTorchIOImage)
    self.setGIFColors(outputLabelMapNode)
    return outputLabelMapNode

  def infer(self, model, torchIOImage, useMixedPrecision):
    # TODO
    from TorchIOUtils import TorchIOUtilsLogic
    tioLogic = TorchIOUtilsLogic()
    threshold = tioLogic.torchio.Lambda(lambda x: x > x.mean())
    output = threshold(torchIOImage)
    return output

  def setGIFColors(self, labelMapVolumeNode):
    pass  # TODO
