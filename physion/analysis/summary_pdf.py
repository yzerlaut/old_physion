import sys, time, tempfile, os, pathlib, json, datetime, string
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# from PyQt5 import QtGui, QtWidgets, QtCore
# import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz.show_data import MultimodalData
from analysis.orientation_direction_selectivity import  orientation_selectivity_analysis, direction_selectivity_analysis

def make_sumary_pdf(filename, Nmax=np.inf,
                    T_raw_data=90, N_raw_data=3, ROI_raw_data=20, Tbar_raw_data=5):

    data = MultimodalData(filename)
    data.roiIndices = np.sort(np.random.choice(np.arange(np.sum(data.iscell)),
                                               size=ROI_raw_data, replace=False))

    
    with PdfPages(filename.replace('nwb', 'pdf')) as pdf:
        
        # # plot raw data sample
        # for t0 in np.linspace(T_raw_data, data.tlim[1], N_raw_data):
        #     TLIM = [np.max([10,t0-T_raw_data]),t0]
        #     fig, ax = plt.subplots(1, figsize=(8.27, 5))
        #     fig.subplots_adjust(top=0.85)
        #     data.plot(TLIM,
        #               settings={'Photodiode':dict(fig_fraction=0.1, subsampling=10, color='grey'),
        #                         'Locomotion':dict(fig_fraction=1, subsampling=10, color='b'),
        #                         'Pupil':dict(fig_fraction=2, subsampling=10, color='red'),
        #                         'CaImagingSum':dict(fig_fraction=2, 
        #                                 quantity='CaImaging', subquantity='Fluorescence', color='green'),
        #                         'CaImaging':dict(fig_fraction=7, 
        #                                          quantity='CaImaging', subquantity='dF/F', vicinity_factor=1., color='tab',
        #                                          roiIndices=data.roiIndices),
        #                         'VisualStim':dict(fig_fraction=0.01, color='black')},
        #               ax=ax, Tbar=Tbar_raw_data)
        #     # inset with time sample
        #     axT = plt.axes([0.6, 0.9, 0.3, 0.05])
        #     axT.axis('off')
        #     axT.plot(data.tlim, [0,0], 'k-', lw=2)
        #     axT.plot(TLIM, [0,0], '-', color=plt.cm.tab10(3), lw=5)
        #     axT.annotate('0 ', (0,0), xycoords='data', ha='right', fontsize=9)
        #     axT.annotate(' %.1fmin' % (data.tlim[1]/60.), (data.tlim[1],0), xycoords='data', fontsize=9)
        #     pdf.savefig()  # saves the current figure into a pdf page
        #     plt.close()

        # looping over protocols
        for p, protocol in enumerate(data.protocols):
            # finding protocol type
            protocol_type = (data.metadata['Protocol-%i-Stimulus' % (p+1)] if (len(data.protocols)>1) else data.metadata['Stimulus'])
            
            # protocol-dependent analysis
            if protocol_type=='full-field-grating':
                for i in range(data.iscell.sum())[:Nmax]:
                    fig, SI, responsive = orientation_selectivity_analysis(data, roiIndex=i)
                    pdf.savefig()  # saves the current figure into a pdf page
            elif protocol_type=='drifting-full-field-grating':
                for i in range(data.iscell.sum())[:Nmax]:
                    fig, SI, responsive = direction_selectivity_analysis(data, roiIndex=i)
                    pdf.savefig()  # saves the current figure into a pdf page
            
        # print(data.protocols)
        # print(data.nwbfile.stimulus)
        # print(data.metadata)

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

    make_sumary_pdf(filename, Nmax=2)        
