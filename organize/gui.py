import sys, time, os, pathlib
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from organize.compress import compress_datafolder
from misc.style import set_dark_style, set_app_icon

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, parent=None):
        """
        sampling in Hz
        """
        super(MainWindow, self).__init__()


        self.setGeometry(30,30,700,700)

def run(app):
    set_dark_style(app)
    set_app_icon(app)
    return MainWindow(app)

if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = run(app)
    sys.exit(app.exec_())
        

