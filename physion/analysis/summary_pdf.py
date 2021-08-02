import sys, time, tempfile, os, pathlib, json, datetime, string, subprocess
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.folders import python_path
from dataviz.show_data import MultimodalData
from Ca_imaging.tools import compute_CaImaging_trace
from analysis.tools import *
from analysis import orientation_direction_selectivity, spont_behavior, rois


def raw_data_plot_settings(data, subsampling_factor=1):
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
                                     roiIndices=np.random.choice(np.arange(np.sum(data.iscell)), 10, replace=True), # picking 20 random non-redundant rois
                                     quantity='CaImaging',
                                     subquantity='dF/F', vicinity_factor=1., color='green', subsampling=subsampling_factor)
        settings['CaImagingRaster'] = dict(fig_fraction=5,
                                           roiIndices='all',
                                           quantity='CaImaging', subquantity='Fluorescence', normalization='per-cell',
                                           subsampling=5*subsampling_factor)
    if not (subsampling_factor>1):
        settings['VisualStim'] = dict(fig_fraction=0.01, color='black')
    return settings


def metadata_fig(data):
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, figsize=(11.4, 3.5))
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    s=''
    for key in ['protocol', 'subject_ID', 'notes']:
        s+='- %s :\n    "%s" \n' % (key, data.metadata[key])
    s += '- completed:\n       n=%i/%i episodes' %(data.nwbfile.stimulus['time_start_realigned'].data.shape[0],
                                                   data.nwbfile.stimulus['time_start'].data.shape[0])
    ax.annotate(s, (0,1), va='top', fontsize=9)
    s=''
    for key in data.metadata['subject_props']:
        s+='- %s :  "%s" \n' % (key, data.metadata['subject_props'][key])
    ax.annotate(s, (0.3,1), va='top', fontsize=8)

    s=''
    for i, key in enumerate(data.metadata):
        s+='- %s :  "%s"' % (key, str(data.metadata[key])[-20:])
        if i%3==2:
            s+='\n'
    ax.annotate(s, (1,1), va='top', ha='right', fontsize=6)
    
    ax.axis('off')

    s, ds ='', 150
    for key in data.nwbfile.devices:
        S = str(data.nwbfile.devices[key])
        # print(S[:100], len(S))
        i=0
        while i<len(S)-ds:
            s += S[i:i+ds]+'\n'
            i+=ds
    ax.annotate(s, (0,0), fontsize=6)
        
    return fig




def make_summary_pdf(filename, Nmax=1000000,
                    include=['exp', 'raw', 'behavior', 'rois', 'protocols'],
                    verbose=True):

    data = MultimodalData(filename)
    
    folder = summary_pdf_folder(filename)
    
    if not os.path.isdir(folder):
        os.mkdir(folder)
        print('-> created summary PDF folder !')
    else:
        print('summary PDF folder "%s" already exists !' % folder)
        
    if 'exp' in include:
        
        with PdfPages(os.path.join(folder, 'exp.pdf')) as pdf:

            print('* writing experimental metadata as "exp.pdf" [...] ')
            
            print('   - notes')
            fig = metadata_fig(data)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
        print('[ok] notes saved as: "%s" ' % os.path.join(folder, 'exp.pdf'))
        

    if 'behavior' in include:
        
        process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'spont_behavior.py')
        p = subprocess.Popen('%s %s %s' % (python_path, process_script, filename), shell=True)

    if 'raw' in include:
        
        process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'raw_data.py')
        p = subprocess.Popen('%s %s %s' % (python_path, process_script, filename), shell=True)
        
                
    if 'rois' in include:
        
        process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'rois.py')
        p = subprocess.Popen('%s %s %s' % (python_path, process_script, filename), shell=True)
        
    if 'protocols' in include:
        print('* looping over protocols for analysis [...]')
        
        # looping over protocols
        for p, protocol in enumerate(data.protocols):

            print('* * plotting protocol "%s" [...]' % protocol)
            
            # with PdfPages(os.path.join(folder, '%s.pdf' % protocol)) as pdf:
            
            #     # finding protocol type
            #     protocol_type = (data.metadata['Protocol-%i-Stimulus' % (p+1)] if (len(data.protocols)>1) else data.metadata['Stimulus'])
            #     # print(protocol_type)
            #     # then protocol-dependent analysis

            #     if protocol_type=='full-field-grating':
            #         Nresp, SIs = 0, []
            #         for i in data.roiIndices[:Nmax]:
            #             print('   - plotting analysis of ROI #%i' % (i+1))
            #             fig, SI, responsive = orientation_direction_selectivity.OS_ROI_analysis(data, roiIndex=i, verbose=False)
            #             pdf.savefig()  # saves the current figure into a pdf page
            #             plt.close()
            #             if responsive:
            #                 Nresp += 1
            #                 SIs.append(SI)
            #         # summary figure for this protocol
            #         fig, AX = orientation_direction_selectivity.summary_fig(Nresp, data.iscell.sum(), np.array(SIs))
            #         pdf.savefig()  # saves the current figure into a pdf page
            #         plt.close()

            #     elif protocol_type=='drifting-full-field-grating':
            #         Nresp, SIs = 0, []
            #         for i in data.roiIndices[:Nmax]:
            #             print('   - plotting analysis of ROI #%i' % (i+1))
            #             fig, SI, responsive = orientation_direction_selectivity.DS_ROI_analysis(data, roiIndex=i, verbose=False)
            #             pdf.savefig()  # saves the current figure into a pdf page
            #             plt.close()
            #             if responsive:
            #                 Nresp += 1
            #                 SIs.append(SI)
            #         fig, AX = orientation_direction_selectivity.summary_fig(Nresp, data.iscell.sum(), np.array(SIs),
            #                                                                 label='Direction Select. Index')
            #         pdf.savefig()  # saves the current figure into a pdf page
            #         plt.close()

            #     elif protocol_type in ['center-grating', 'drifting-center-grating']:
            #         from analysis.surround_suppression import orientation_size_selectivity_analysis
            #         from analysis.orientation_direction_selectivity import summary_fig
            #         Nresp, SIs = 0, []
            #         for i in range(data.iscell.sum())[:Nmax]:
            #             fig, responsive = orientation_size_selectivity_analysis(data,
            #                                                     roiIndex=i, verbose=False)
            #             pdf.savefig()  # saves the current figure into a pdf page
            #             plt.close()
            #             if responsive:
            #                 Nresp += 1
            #                 SIs.append(0) # TO BE FILLED
            #         fig, AX = summary_fig(Nresp, data.iscell.sum(), np.array(SIs),
            #                               label='none')
            #         pdf.savefig()  # saves the current figure into a pdf page
            #         plt.close()

            #     elif 'noise' in protocol_type:
            #         from receptive_field_mapping import RF_analysis
            #         Nresp, SIs = 0, []
            #         for i in range(data.iscell.sum())[:Nmax]:
            #             fig, SI, responsive = RF_analysis(data, roiIndex=i, verbose=False)
            #             pdf.savefig()  # saves the current figure into a pdf page
            #             plt.close()
            #             if responsive:
            #                 Nresp += 1
            #                 SIs.append(0) # TO BE FILLED
            #         fig, AX = summary_fig(Nresp, data.iscell.sum(), np.array(SIs),
            #                               label='none')
            #         pdf.savefig()  # saves the current figure into a pdf page
            #         plt.close()

            #     elif 'spatial-location' in protocol_type:
            #         from surround_suppression import orientation_size_selectivity_analysis
            #         Nresp, SIs = 0, []
            #         for i in range(data.iscell.sum())[:Nmax]:
            #             fig, responsive = orientation_size_selectivity_analysis(data, roiIndex=i, verbose=False)
            #             pdf.savefig()  # saves the current figure into a pdf page
            #             plt.close()
            #             if responsive:
            #                 Nresp += 1
            #                 SIs.append(0) # TO BE FILLED
            #         fig, AX = summary_fig(Nresp, data.iscell.sum(), np.array(SIs),
            #                               label='none')
            #         pdf.savefig()  # saves the current figure into a pdf page
            #         plt.close()


    print('subprocesses to analyze "%s" were launched !' % filename)


if __name__=='__main__':
    
    import argparse, datetime

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str,
                        default='/home/yann/DATA/CaImaging/NDNFcre_GCamp6s/2021_07_01/2021_07_01-16-27-22.nwb')
    parser.add_argument('-o', "--ops", type=str, nargs='*',
                        default=['exp', 'raw', 'rois', 'behavior'])
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    
    args = parser.parse_args()

    make_summary_pdf(args.datafile,
                     include=args.ops,
                     Nmax=args.Nmax,
                     verbose=args.verbose)

    







