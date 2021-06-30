from contextlib import contextmanager

import qt
import slicer


def getPythonConsoleWidget():
  return slicer.util.mainWindow().pythonConsole().parent()

@contextmanager
def peakPythonConsole(show=True):
  if show:
    console = getPythonConsoleWidget()
    pythonVisible = console.visible
    console.setVisible(True)
  try:
    yield
  finally:
    if show:
      console.setVisible(pythonVisible)


@contextmanager
def showWaitCursor(show=True):
  if show:
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
  try:
    yield
  finally:
    if show:
      qt.QApplication.restoreOverrideCursor()


@contextmanager
def messageContextManager(message):
  box = qt.QMessageBox()
  box.setStandardButtons(0)
  box.setText(message)
  box.show()
  slicer.app.processEvents()
  try:
    yield
  finally:
    box.accept()
