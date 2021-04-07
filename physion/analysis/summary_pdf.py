import sys, time, tempfile, os, pathlib, json, datetime, string
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# from PyQt5 import QtGui, QtWidgets, QtCore
# import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz.show_data import MultimodalData
from analysis.orientation_direction_selectivity import  orientation_selectivity_analysis, direction_selectivity_analysis

def make_sumary_pdf(filename, Nmax=1000000,
                    T_raw_data=90, N_raw_data=3, ROI_raw_data=15, Tbar_raw_data=5):

    data = MultimodalData(filename)
    data.roiIndices = np.sort(np.random.choice(np.arange(np.sum(data.iscell)),
                                               size=ROI_raw_data, replace=False))

    
    with PdfPages(filename.replace('nwb', 'pdf')) as pdf:
        
        # plot raw data sample
        for t0 in np.linspace(T_raw_data, data.tlim[1], N_raw_data):
            TLIM = [np.max([10,t0-T_raw_data]),t0]
            fig, ax = plt.subplots(1, figsize=(11.4, 5))
            fig.subplots_adjust(top=0.8, bottom=0.05)
            data.plot(TLIM,
                      settings={'Photodiode':dict(fig_fraction=0.1, subsampling=10, color='grey'),
                                'Locomotion':dict(fig_fraction=1, subsampling=10, color='b'),
                                'Pupil':dict(fig_fraction=2, subsampling=10, color='red'),
                                'CaImagingSum':dict(fig_fraction=2, 
                                        quantity='CaImaging', subquantity='Fluorescence', color='green'),
                                'CaImaging':dict(fig_fraction=7, 
                                                 quantity='CaImaging', subquantity='dF/F', vicinity_factor=1., color='tab',
                                                 roiIndices=data.roiIndices),
                                'VisualStim':dict(fig_fraction=0.01, color='black')},
                      ax=ax, Tbar=Tbar_raw_data)
            # inset with time sample
            axT = plt.axes([0.6, 0.9, 0.3, 0.05])
            axT.axis('off')
            axT.plot(data.tlim, [0,0], 'k-', lw=2)
            axT.plot(TLIM, [0,0], '-', color=plt.cm.tab10(3), lw=5)
            axT.annotate('0 ', (0,0), xycoords='data', ha='right', fontsize=9)
            axT.annotate(' %.1fmin' % (data.tlim[1]/60.), (data.tlim[1],0), xycoords='data', fontsize=9)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

        print('looping over protocols and cells [...]')
        
        # looping over protocols
        for p, protocol in enumerate(data.protocols):
            # finding protocol type
            protocol_type = (data.metadata['Protocol-%i-Stimulus' % (p+1)] if (len(data.protocols)>1) else data.metadata['Stimulus'])
            # then protocol-dependent analysis
            
            if protocol_type=='full-field-grating':
                Nresp, SIs = 0, []
                for i in range(data.iscell.sum())[:Nmax]:
                    fig, SI, responsive = orientation_selectivity_analysis(data, roiIndex=i, verbose=False)
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()
                    if responsive:
                        Nresp += 1
                        SIs.append(SI)
                # summary figure for this protocol
                fig, AX = summary_fig(Nresp, data.iscell.sum(), np.array(SIs))
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
                
            elif protocol_type=='drifting-full-field-grating':
                Nresp, SIs = 0, []
                for i in range(data.iscell.sum())[:Nmax]:
                    fig, SI, responsive = direction_selectivity_analysis(data, roiIndex=i, verbose=False)
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()
                    if responsive:
                        Nresp += 1
                        SIs.append(SI)
                fig, AX = summary_fig(Nresp, data.iscell.sum(), np.array(SIs))
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
            
        print('[ok] pdf succesfully saved as "%s" !' % filename.replace('nwb', 'pdf'))        


def summary_fig(Nresp, Ntot, quantity,
                label='Orient. Select. Index'):
    fig, AX = plt.subplots(1, 4, figsize=(11.4, 2.5))
    fig.subplots_adjust(left=0.1, right=0.8, bottom=0.2)
    AX[1].pie([100*Nresp/Ntot, 100*(1-Nresp/Ntot)], explode=(0, 0.1), colors=[plt.cm.tab10(3), plt.cm.tab10(2)],
              labels=['unresponsive', 'responsive'], autopct='%1.1f%%', shadow=True, startangle=90)
    AX[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    AX[3].hist(quantity)
    AX[3].set_xlabel(label, fontsize=9)
    AX[3].set_ylabel('count', fontsize=9)
    for ax in [AX[0], AX[2]]:
        ax.axis('off')
    return fig, AX
    

    
if __name__=='__main__':
    
    filename = '/home/yann/DATA/Wild_Type/2021_03_11-17-13-03.nwb'

    make_sumary_pdf(filename)
    
    # fig, AX = summary_fig(20, 100, np.random.randn(100))
    # plt.show()
    
    
