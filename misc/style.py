import os
from PyQt5 import QtGui, QtCore, QtWidgets

def set_dark_style(win):

    win.setStyleSheet("QMainWindow {background: 'black';}")
    win.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                               "color:white;}")
    win.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(100,50,100); "
                             "color:white;}")
    win.styleInactive = ("QPushButton {Text-align: left; "
                              "background-color: rgb(50,50,50); "
                              "color:gray;}")
    

def set_app_icon(app):
    icon_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'doc', "icon.png")
    app_icon = QtGui.QIcon()
    for x in [16, 24, 32, 40, 96, 256]:
        app_icon.addFile(icon_path, QtCore.QSize(x, x))
    
    
