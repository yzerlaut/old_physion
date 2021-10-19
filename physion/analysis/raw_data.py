import sys, os, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz.show_data import MultimodalData
from dataviz import tools
from Ca_imaging.tools import compute_CaImaging_trace
from analysis.tools import *


def raw_data_plot_settings(data,
                           Nroi=10,
                           subsampling_factor=1):
    settings = {}

    if 'Photodiode-Signal' in data.nwbfile.acquisition:
        settings['Photodiode'] = dict(fig_fraction=0.1, subsampling=100*subsampling_factor, color='grey')
    if 'Running-Speed' in data.nwbfile.acquisition:
        settings['Locomotion'] = dict(fig_fraction=2, subsampling=3*subsampling_factor)
    if 'Pupil' in data.nwbfile.processing:
        settings['Pupil'] = dict(fig_fraction=2, subsampling=2*subsampling_factor)
    if 'FaceMotion' in data.nwbfile.processing:
        settings['FaceMotion'] = dict(fig_fraction=2, subsampling=2*subsampling_factor)
    if 'ophys' in data.nwbfile.processing:
        settings['CaImaging'] = dict(fig_fraction=5,
                                     roiIndices=np.random.choice(np.arange(np.sum(data.iscell)), Nroi, replace=True), # picking 20 random non-redundant rois
                                     quantity='CaImaging',
                                     subquantity='dF/F', vicinity_factor=1., color='green', subsampling=subsampling_factor)
        settings['CaImagingRaster'] = dict(fig_fraction=5,
                                           roiIndices='all',
                                           quantity='CaImaging', subquantity='Fluorescence', normalization='per-cell',
                                           subsampling=5*subsampling_factor)
    for signal in ['Electrophysiological-Signal', 'LFP', 'Vm']:
        if signal in data.nwbfile.acquisition:
            settings['Electrophy'] = dict(fig_fraction=1,
                                          color='blue', subsampling=10*subsampling_factor)
        
    if not (subsampling_factor>1):
        settings['VisualStim'] = dict(fig_fraction=0.01, color='black')
    return settings


def analysis_pdf(datafile,
                 NzoomPlot=5,
                 Nroi=10):

    pdf_filename = os.path.join(summary_pdf_folder(datafile), 'raw.pdf')

    data = MultimodalData(datafile)

    if data.metadata['CaImaging']:
        Tzoom=60
    else:
        Tzoom=20

    with PdfPages(pdf_filename) as pdf:
        
        print('* plotting raw data as "raw.pdf" [...]')
        
        fig, ax = plt.subplots(1, figsize=(11.4, 5))
        fig.subplots_adjust(top=0.8, bottom=0.05)
        
        print('   - plotting full data view')
        if data.metadata['CaImaging']:
            subsampling = max([int((data.tlim[1]-data.tlim[0])/data.CaImaging_dt/1000), 1])
        else:
            subsampling = 1
            
        data.plot_raw_data(data.tlim,
                           settings=raw_data_plot_settings(data,
                                                           Nroi=Nroi,
                                                           subsampling_factor=int((data.tlim[1]-data.tlim[0])/60.+1)),
                           ax=ax)
        axT = add_inset_with_time_sample(data.tlim, data.tlim, plt)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # plot raw data sample
        for t0 in np.linspace(Tzoom, data.tlim[1], NzoomPlot):
            
            TLIM = [np.max([10,t0-Tzoom]),t0]
            print('   - plotting raw data sample at times ', TLIM)
            
            fig, ax = plt.subplots(1, figsize=(11.4, 5))
            fig.subplots_adjust(top=0.8, bottom=0.05)
            data.plot_raw_data(TLIM, settings=raw_data_plot_settings(data, Nroi=Nroi), ax=ax)
            axT = add_inset_with_time_sample(TLIM, data.tlim, plt)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
                
    print('[ok] raw data plot saved as: "%s" ' % pdf_filename)

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    # parser.add_argument("--Tzoom", type=float, default=60.)
    parser.add_argument("--NzoomPlot", type=int, default=5)
    parser.add_argument("--Nroi", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    
    if '.nwb' in args.datafile:
        analysis_pdf(args.datafile,
                     # Tzoom=args.Tzoom,
                     NzoomPlot=args.NzoomPlot,
                     Nroi=args.Nroi)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')








