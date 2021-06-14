import sys, time, tempfile, os, pathlib, json, datetime, string
from PyQt5 import QtGui, QtWidgets, QtCore
import numpy as np
import pyqtgraph as pg
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder
from misc.guiparts import NewWindow
from scipy.interpolate import interp1d
from misc.colors import build_colors_from_array

class BehavioralModWindow(NewWindow):

    def __init__(self,
                 app,
                 parent=None,
                 dataset=None,
                 dt_sampling=10, # ms
                 title='Behavioral-Modulation'):

        super(BehavioralModWindow, self).__init__(parent=parent,
                                                  title=title)

        self.show()
