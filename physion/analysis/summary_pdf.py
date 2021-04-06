import sys, time, tempfile, os, pathlib, json, datetime, string
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# from PyQt5 import QtGui, QtWidgets, QtCore
# import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz.show_data import MultimodalData


def make_sumary_pdf(filename):

    data = MultimodalData(filename)
    data.roiIndices = np.sort(np.random.choice(np.arange(np.sum(data.iscell)), size=20, replace=False))
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = plt.subplot(111)
    data.plot([10,200],
              settings={'Photodiode':dict(fig_fraction=0.1, subsampling=10, color='grey'),
                        'Locomotion':dict(fig_fraction=1, subsampling=10, color='b'),
                        'Pupil':dict(fig_fraction=2, subsampling=10, color='red'),
                        'CaImagingSum':dict(fig_fraction=2, 
                                            quantity='CaImaging', subquantity='Fluorescence', color='green'),
                        'CaImaging':dict(fig_fraction=5, 
                                         quantity='CaImaging', subquantity='Fluorescence', color='green',
                                         roiIndices=data.roiIndices[:10], vicinity_factor=1.5),
                        'VisualStim':dict(fig_fraction=0.01, color='black')},                    
              ax=ax)
    plt.show()
    # with PdfPages(filename.replace('nwb', 'pdf')) as pdf:
    #     # plt raw data
        
    #     fig = plt.figure(figsize=(11.69, 8.27))
    #     ax = plt.subplot(111)
    #     data.plot([0,100], ax=ax)
    #     # plt.subplots_adjust(left=0, right=1, top=1, bottom=0.1)
    #     # plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
    #     # plt.title('Page One')
    #     pdf.savefig()  # saves the current figure into a pdf page
    #     plt.close()

    #     # # if LaTeX is not installed or error caught, change to `False`
    #     # plt.rcParams['text.usetex'] = True
    #     # plt.figure(figsize=(8, 6))
    #     # x = np.arange(0, 5, 0.1)
    #     # plt.plot(x, np.sin(x), 'b-')
    #     # plt.title('Page Two')
    #     # pdf.attach_note("plot of sin(x)")  # attach metadata (as pdf note) to page
    #     # pdf.savefig()
    #     # plt.close()

    #     # plt.rcParams['text.usetex'] = False
    #     # fig = plt.figure(figsize=(4, 5))
    #     # plt.plot(x, x ** 2, 'ko')
    #     # plt.title('Page Three')
    #     # pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    #     # plt.close()

    #     # We can also set the file's metadata via the PdfPages object:
    #     d = pdf.infodict()
    #     d['Title'] = filename
    #     d['Author'] = 'Jouni K. Sepp\xe4nen'
    #     d['Subject'] = 'How to create a multipage pdf file and set its metadata'
    #     d['Keywords'] = 'PdfPages multipage keywords author title subject'
    #     d['CreationDate'] = datetime.datetime.today()

        
if __name__=='__main__':
    
    filename = '/home/yann/DATA/Wild_Type/2021_03_11-17-13-03.nwb'

    make_sumary_pdf(filename)        
