import sys, os, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from datavyz import graph_env_manuscript as ge

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz.show_data import MultimodalData
from dataviz import tools
from Ca_imaging.tools import compute_CaImaging_trace
from analysis.tools import *


def raw_fluo_fig(data, roiIndex=0, t_zoom=[0,30]):

    MODALITIES, QUANTITIES, TIMES, UNITS = find_modalities(data)

    plt.style.use('ggplot')
    fig, AX = plt.subplots(8, 1, figsize=(11.4, 7))
    plt.subplots_adjust(left=0.08, right=0.93, bottom=0.3/(2+len(MODALITIES)), top=0.95, wspace=0, hspace=0.2)

    AX[0].annotate('\n   ROI#%i' % (roiIndex+1), (0., 1.), xycoords='figure fraction', weight='bold', fontsize=11, va='top')
    
    _, Fmin = compute_CaImaging_trace(data, 'dF/F', [roiIndex], with_baseline=True)
    
    for tlim, ax1, ax2 in zip([data.tlim, data.tlim[0]+10+np.array(t_zoom), data.tlim[1]-10-np.array(t_zoom)[::-1]],
                              [AX[1], AX[4], AX[7]], [AX[0], AX[3], AX[6]]):
        subsampling = max([int((tlim[1]-tlim[0])/data.CaImaging_dt/1000), 1])
        i1, i2 = tools.convert_times_to_indices(*tlim, data.Neuropil, axis=1)
        tt = np.array(data.Neuropil.timestamps[:])[np.arange(i1,i2)][::subsampling]
        ax1.plot(tt, data.Fluorescence.data[data.validROI_indices[roiIndex], :][np.arange(i1,i2)][::subsampling], color='green', label='raw F')
        ax1.plot(tt, data.Neuropil.data[data.validROI_indices[roiIndex], :][np.arange(i1,i2)][::subsampling], label='Neuropil')
        ax1.plot(tt, Fmin[0][np.arange(i1,i2)][::subsampling], label='F0')
        ax1.set_xlim([data.shifted_start(tlim)-0.01*(tlim[1]-tlim[0]),tlim[1]+0.01*(tlim[1]-tlim[0])])
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('fluo.')
        ax1.legend(loc=(.98,0.4), fontsize=7)

        # reporting average quantities
        ax2.set_title('<F>=%.1f, <F0>=%.1f, <Neuropil>=%.1f' % (\
                                                            np.mean(data.Fluorescence.data[data.validROI_indices[roiIndex], :][np.arange(i1,i2)]),
                                                            np.mean(Fmin[0][np.arange(i1,i2)][::subsampling]),
                                                            np.mean(data.Neuropil.data[data.validROI_indices[roiIndex], :][np.arange(i1,i2)])),
                      fontsize=7)
        
        settings = {}
        settings['Locomotion'] = dict(fig_fraction=1, subsampling=subsampling, color='b')
        if 'Facemotion' in data.nwbfile.processing:
            settings['FaceMotion'] = dict(fig_fraction=1, subsampling=subsampling, color='purple')
        if 'Pupil' in data.nwbfile.processing:
            settings['Pupil'] = dict(fig_fraction=2, subsampling=subsampling, color='red')
        settings['CaImaging'] = dict(fig_fraction=4, subsampling=subsampling,
                                    quantity='CaImaging', subquantity='dF/F', color='green',
                                    roiIndices=[roiIndex], name='')
                 
        data.plot_raw_data(tlim=tlim, settings=settings, ax=ax2, Tbar=int((tlim[1]-tlim[0])/30.))
    
    for i in [2,5]:
        AX[i].axis('off')
    

def analysis_fig(data, roiIndex=0):
    

    MODALITIES, QUANTITIES, TIMES, UNITS = find_modalities(data)
    
    plt.style.use('ggplot')
    
    fig, AX = plt.subplots(2+len(MODALITIES), 5, figsize=(11.4, 2.3*(2+len(MODALITIES))))
    
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.3/(2+len(MODALITIES)), top=0.94, wspace=.5, hspace=.5)
    if len(MODALITIES)==0:
        AX = [AX]
    index = np.arange(len(data.iscell))[data.iscell][roiIndex]
    
    AX[0][0].annotate('ROI#%i' % (roiIndex+1), (0.5, 0.5), weight='bold', fontsize=10, va='center', ha='center')
    AX[0][0].axis('off')
    AX[0][-1].axis('off')
    
    data.show_CaImaging_FOV(key='meanImgE', cmap='viridis', ax=AX[0][1], roiIndex=roiIndex)
    data.show_CaImaging_FOV(key='meanImgE', cmap='viridis', ax=AX[0][2], roiIndex=roiIndex, with_roi_zoom=True)
    try:
        data.show_CaImaging_FOV(key='meanImg_chan2', cmap='viridis', ax=AX[0][3], roiIndex=roiIndex, with_roi_zoom=True)
        AX[0][3].annotate('red cell: %s' % ('yes' if data.redcell[index] else 'no'), (0.5, 0.), xycoords='axes fraction', fontsize=8, va='top', ha='center')
    except KeyError:
        AX[0][3].axis('off')
        AX[0][3].annotate('no red channel', (0.5, 0.5), xycoords='axes fraction', fontsize=9, va='center', ha='center')

    dFoF = compute_CaImaging_trace(data, 'dF/F', [roiIndex]).sum(axis=0) # valid ROI indices inside
    
    AX[1][0].hist(data.Fluorescence.data[index, :], bins=30,
                  weights=100*np.ones(len(dFoF))/len(dFoF))
    AX[1][0].set_xlabel('Fluo. (a.u.)', fontsize=10)
    
    AX[1][1].hist(dFoF, bins=30,
                  weights=100*np.ones(len(dFoF))/len(dFoF))
    AX[1][1].set_xlabel('dF/F', fontsize=10)
    
    AX[1][2].hist(dFoF, log=True, bins=30,
                  weights=100*np.ones(len(dFoF))/len(dFoF))
    AX[1][2].set_xlabel('dF/F', fontsize=10)
    for ax in AX[1][:3]:
        ax.set_ylabel('occurence (%)', fontsize=10)
                  
    CC, ts = autocorrel_on_NWB_quantity(Q1=None, q1=dFoF, t_q1=data.Neuropil.timestamps[:], tmax=180)
    AX[1][3].plot(ts/60., CC, '-', lw=2)
    AX[1][3].set_xlabel('time (min)', fontsize=10)
    AX[1][3].set_ylabel('auto correl.', fontsize=10)

    CC, ts = autocorrel_on_NWB_quantity(Q1=None, q1=dFoF, t_q1=data.Neuropil.timestamps[:], tmax=10)
    AX[1][4].plot(ts, CC, '-', lw=2)
    AX[1][4].set_xlabel('time (s)', fontsize=10)
    AX[1][4].set_ylabel('auto correl.', fontsize=10)

    for i, mod, quant, times, unit in zip(range(len(TIMES)), MODALITIES, QUANTITIES, TIMES, UNITS):
        
        AX[2+i][0].set_title(mod, fontsize=10, color=plt.cm.tab10(i))
        
        if times is None:
            Q, qq = quant, None
        else:
            Q, qq = None, quant

        hist, be1, be2 = hist2D_on_NWB_quantity(Q1=Q, Q2=None,
                    q1=qq, t_q1=times, q2=dFoF, t_q2=data.Neuropil.timestamps[:], bins=50)
        hist = np.log(np.clip(hist, np.min(hist[hist>0]), np.max(hist)))
        ge.matrix(hist, x=be1, y=be2, colormap=plt.cm.binary, ax=AX[2+i][0], aspect='auto')
        AX[2+i][0].grid(False)
        AX[2+i][0].set_ylabel(unit, fontsize=8)
        AX[2+i][0].set_xlabel('dF/F', fontsize=8)
        ge.annotate(AX[2+i][0], '  log distrib.', (0,1), va='top', size='x-small')
            
        mean_q1, var_q1, mean_q2, var_q2 = crosshistogram_on_NWB_quantity(Q1=Q, Q2=None,
                                q1=qq, t_q1=times, q2=dFoF, t_q2=data.Neuropil.timestamps[:], Npoints=30)
        
        AX[2+i][1].errorbar(mean_q1, mean_q2, xerr=var_q1, yerr=var_q2, color=plt.cm.tab10(i))
        AX[2+i][1].set_xlabel(unit, fontsize=10)
        AX[2+i][1].set_ylabel('dF/F', fontsize=10)

        mean_q1, var_q1, mean_q2, var_q2 = crosshistogram_on_NWB_quantity(Q2=Q, Q1=None,
                                q2=qq, t_q2=times, q1=dFoF, t_q1=data.Neuropil.timestamps[:], Npoints=30)
        
        AX[2+i][2].errorbar(mean_q1, mean_q2, xerr=var_q1, yerr=var_q2, color=plt.cm.tab10(i))
        AX[2+i][2].set_ylabel(unit, fontsize=10)
        AX[2+i][2].set_xlabel('dF/F', fontsize=10)
        
        CCF, tshift = crosscorrel_on_NWB_quantity(Q1=Q, Q2=None,
                                q1=qq, t_q1=times, q2=dFoF, t_q2=data.Neuropil.timestamps[:], tmax=180)
        AX[2+i][3].plot(tshift/60, CCF, '-', color=plt.cm.tab10(i))
        AX[2+i][3].set_xlabel('time (min)', fontsize=10)
        AX[2+i][3].set_ylabel('cross correl.', fontsize=10)

        CCF, tshift = crosscorrel_on_NWB_quantity(Q1=Q, Q2=None,
                         q1=qq, t_q1=times, q2=dFoF, t_q2=data.Neuropil.timestamps[:], tmax=20)
        AX[2+i][4].plot(tshift, CCF, '-', color=plt.cm.tab10(i))
        AX[2+i][4].set_xlabel('time (s)', fontsize=10)
        AX[2+i][4].set_ylabel('cross correl.', fontsize=10)
        

    return fig


def analysis_pdf(datafile, Nmax=1000000):

    pdf_filename = os.path.join(summary_pdf_folder(datafile), 'rois.pdf')

    data = MultimodalData(datafile)
    
    with PdfPages(pdf_filename) as pdf:

        print('* plotting ROI analysis as "rois.pdf" [...]')
        print('   - plotting imaging FOV')
        fig, AX = plt.subplots(1, 5, figsize=(11.4, 2.5))
        plt.subplots_adjust(left=0.03, right=0.97)
        data.show_CaImaging_FOV(key='meanImg', NL=1, cmap='viridis', ax=AX[0])
        data.show_CaImaging_FOV(key='meanImg', NL=2, cmap='viridis', ax=AX[1])
        data.show_CaImaging_FOV(key='meanImgE', NL=2, cmap='viridis', ax=AX[2])
        data.show_CaImaging_FOV(key='max_proj', NL=2, cmap='viridis', ax=AX[3])
        try:
            data.show_CaImaging_FOV(key='meanImg_chan2', NL=2, cmap='viridis', ax=AX[4])
        except KeyError:
            AX[4].annotate('no red channel', (0.5, 0.5), xycoords='axes fraction', fontsize=9, va='center', ha='center')
            AX[4].axis('off')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
        for i in np.arange(data.iscell.sum())[:Nmax]:
            print('   - plotting analysis of ROI #%i' % (i+1))
            try:
                fig = raw_fluo_fig(data, roiIndex=i)
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
                fig = analysis_fig(data, roiIndex=i)
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
            except ValueError:
                print('  /!\ Pb with ROI #%i /!\ ' % (i+1))

    print('[ok] roi analysis saved as: "%s" ' % pdf_filename)
    
if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if '.nwb' in args.datafile:
        analysis_pdf(args.datafile, Nmax=args.Nmax)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')








