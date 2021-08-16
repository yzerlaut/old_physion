import sys, os, pathlib
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from datavyz import graph_env_manuscript as ge

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz.show_data import MultimodalData, format_key_value
from analysis.tools import summary_pdf_folder
from Ca_imaging.tools import compute_CaImaging_trace

def ROI_analysis(FullData,
                 roiIndex=0,
                 subquantity='Deconvolved',
                 metrics='mean',
                 iprotocol=0,
                 verbose=False):


    dF = compute_CaImaging_trace(FullData, subquantity, [roiIndex]).sum(axis=0)
    t = FullData.Fluorescence.timestamps[:]
    Nall = FullData.nwbfile.stimulus['time_start_realigned'].num_samples
    
    # compute protcol cond
    Pcond = FullData.get_protocol_cond(iprotocol)[:Nall]

    # do REVERSE CORRELATION analysis
    full_img, cum_weight = np.zeros(FullData.visual_stim.screen['resolution'], dtype=float).T, 0

    for i in np.arange(Nall)[Pcond]:
        tstart = FullData.nwbfile.stimulus['time_start_realigned'].data[i]
        tstop = FullData.nwbfile.stimulus['time_stop_realigned'].data[i]
        cond = (t>tstart) & (t<tstop)
        weight = np.inf
        if metrics == 'mean':
            weight = np.mean(dF[cond])
        elif np.sum(cond)>0 and (metrics=='max'):
            weight = np.max(dF[cond])
        if np.isfinite(weight):
            full_img += weight*2*(FullData.visual_stim.get_image(i)-.5)
            cum_weight += weight
        else:
            print('For episode #%i in t=(%.1f, %.1f), pb in estimating the weight !' % (i, tstart, tstop) )

    full_img /= cum_weight

    # GAUSSIAN FILTERING
    # img = gaussian_filter(full_img, (10,10))
    
    fig, ax = ge.figure(figsize=(1.7,1.7))

    ax.imshow(full_img, cmap=plt.cm.PiYG,
              interpolation=None,
              origin='lower',
              aspect='equal',
              extent=[FullData.visual_stim.x.min(), FullData.visual_stim.x.max(),
                      FullData.visual_stim.z.min(), FullData.visual_stim.z.max()],
              vmin=-np.max(np.abs(full_img)), vmax=np.max(np.abs(full_img)))

    ge.set_plot(ax, xlabel='x ($^{o}$)', ylabel='y ($^{o}$)', title='roi #%i' % (roiIndex+1))
    
    return fig
    

def analysis_pdf(datafile, iprotocol=0, Nmax=1000000):

    data = MultimodalData(datafile, with_visual_stim=True)

    pdf_filename = os.path.join(summary_pdf_folder(datafile), '%s-RF_mapping.pdf' % data.protocols[iprotocol])
    
    curves, results = [], {'Ntot':data.iscell.sum()}
    with PdfPages(pdf_filename) as pdf:

        for roi in np.arange(data.iscell.sum())[:Nmax]:
            
            print('   - RF mapping analysis for ROI #%i / %i' % (roi+1, data.iscell.sum()))
            fig = ROI_analysis(data, roiIndex=roi, iprotocol=iprotocol)
            pdf.savefig(fig)
            plt.close(fig)

    print('[ok] RF mapping analysis saved as: "%s" ' % pdf_filename)

if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument("--iprotocol", type=int, default=0, help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    
    if '.nwb' in args.datafile:
        analysis_pdf(args.datafile, iprotocol=args.iprotocol, Nmax=args.Nmax)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')
