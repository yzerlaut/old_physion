import sys, time, tempfile, os, pathlib, json, datetime, string
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz.show_data import MultimodalData
from Ca_imaging.tools import compute_CaImaging_trace
from analysis.tools import *
from analysis import orientation_direction_selectivity

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
        settings['CaImaging'] = dict(fig_fraction=8,
                                     roiIndices=np.random.choice(np.arange(np.sum(data.iscell)), 20, replace=True), # picking 20 random non-redundant rois
                                     quantity='CaImaging',
                                     subquantity='dF/F', vicinity_factor=1., color='green', subsampling=subsampling_factor)
        # settings['CaImagingSum'] = dict(fig_fraction=2,
        #                                 subsampling=10, 
        #                                 quantity='CaImaging', subquantity='dF/F', color='green')
    if not (subsampling_factor>1):
        settings['VisualStim'] = dict(fig_fraction=0.01, color='black')
    return settings

def find_modalities(data):

    MODALITIES, QUANTITIES, TIMES, UNITS = [], [], [], []
    if 'Running-Speed' in data.nwbfile.acquisition:
        MODALITIES.append('Running-Speed')
        QUANTITIES.append(data.nwbfile.acquisition['Running-Speed'])
        TIMES.append(None)
        UNITS.append('cm/s')
    if 'Pupil' in data.nwbfile.processing:
        MODALITIES.append('Pupil')
        area=np.pi*data.nwbfile.processing['Pupil'].data_interfaces['sx'].data[:]*\
            data.nwbfile.processing['Pupil'].data_interfaces['sy'].data[:]
        QUANTITIES.append(area)
        TIMES.append(data.nwbfile.processing['Pupil'].data_interfaces['sy'].timestamps[:])
        UNITS.append('mm$^2$')
    if 'Whisking' in data.nwbfile.processing:
        MODALITIES.append('Whisking')
        
    return MODALITIES, QUANTITIES, TIMES, UNITS
    

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


def roi_analysis_fig(data, roiIndex=0):
    

    MODALITIES, QUANTITIES, TIMES, UNITS = find_modalities(data)
    
    plt.style.use('ggplot')
    fig, AX = plt.subplots(2+len(MODALITIES), 4, figsize=(11.4, 2.3*(2+len(MODALITIES))))
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.3/(2+len(MODALITIES)), top=0.98, wspace=.5, hspace=.5)
    if len(MODALITIES)==0:
        AX = [AX]
    AX[0][0].annotate(' ROI#%i' % (roiIndex+1), (0.4, 0.5), xycoords='axes fraction', weight='bold', fontsize=11, ha='center')
    AX[0][0].axis('off')

    data.show_CaImaging_FOV(key='meanImgE', cmap='viridis', ax=AX[0][1], roiIndex=roiIndex)
    data.show_CaImaging_FOV(key='meanImgE', cmap='viridis', ax=AX[0][2], roiIndex=roiIndex, with_roi_zoom=True)

    dFoF = compute_CaImaging_trace(data, 'dF/F', [roiIndex]).sum(axis=0) # valid ROI indices inside

    index = np.arange(len(data.iscell))[data.iscell][roiIndex]
    AX[0][3].hist(data.Fluorescence.data[index, :], bins=30,
                  weights=100*np.ones(len(dFoF))/len(dFoF), color=plt.cm.tab10(1))
    AX[0][3].set_xlabel('Fluo. (a.u.)', fontsize=10)
    AX[1][0].hist(dFoF, bins=30,
                  weights=100*np.ones(len(dFoF))/len(dFoF))
    AX[1][0].set_xlabel('dF/F', fontsize=10)
    AX[1][1].hist(dFoF, log=True, bins=30,
                  weights=100*np.ones(len(dFoF))/len(dFoF))
    AX[1][1].set_xlabel('dF/F', fontsize=10)
    for ax in AX[1][:3]:
        ax.set_ylabel('occurence (%)', fontsize=10)
                  
    CC, ts = autocorrel_on_NWB_quantity(Q1=None, q1=dFoF, t_q1=data.Neuropil.timestamps[:], tmax=180)
    AX[1][2].plot(ts/60., CC, '-', lw=2)
    AX[1][2].set_xlabel('time (min)', fontsize=10)
    AX[1][2].set_ylabel('auto correl.', fontsize=10)

    CC, ts = autocorrel_on_NWB_quantity(Q1=None, q1=dFoF, t_q1=data.Neuropil.timestamps[:], tmax=10)
    AX[1][3].plot(ts, CC, '-', lw=2)
    AX[1][3].set_xlabel('time (s)', fontsize=10)
    AX[1][3].set_ylabel('auto correl.', fontsize=10)

    for i, mod, quant, times, unit in zip(range(len(TIMES)), MODALITIES, QUANTITIES, TIMES, UNITS):
        
        AX[2+i][0].set_title(mod+40*' ', fontsize=10, color=plt.cm.tab10(i))
        
        if times is None:
            Q, qq = quant, None
        else:
            Q, qq = None, quant

        mean_q1, var_q1, mean_q2, var_q2 = crosshistogram_on_NWB_quantity(Q1=Q, Q2=None,
                                q1=qq, t_q1=times, q2=dFoF, t_q2=data.Neuropil.timestamps[:], Npoints=30)
        
        AX[2+i][0].errorbar(mean_q1, mean_q2, xerr=var_q1, yerr=var_q2, color=plt.cm.tab10(i))
        AX[2+i][0].set_xlabel(unit, fontsize=10)
        AX[2+i][0].set_ylabel('dF/F', fontsize=10)

        mean_q1, var_q1, mean_q2, var_q2 = crosshistogram_on_NWB_quantity(Q2=Q, Q1=None,
                                q2=qq, t_q2=times, q1=dFoF, t_q1=data.Neuropil.timestamps[:], Npoints=30)
        
        AX[2+i][1].errorbar(mean_q1, mean_q2, xerr=var_q1, yerr=var_q2, color=plt.cm.tab10(i))
        AX[2+i][1].set_ylabel(unit, fontsize=10)
        AX[2+i][1].set_xlabel('dF/F', fontsize=10)
        
        CCF, tshift = crosscorrel_on_NWB_quantity(Q1=Q, Q2=None,
                                q1=qq, t_q1=times, q2=dFoF, t_q2=data.Neuropil.timestamps[:], tmax=180)
        AX[2+i][2].plot(tshift/60, CCF, '-', color=plt.cm.tab10(i))
        AX[2+i][2].set_xlabel('time (min)', fontsize=10)
        AX[2+i][2].set_ylabel('cross correl.', fontsize=10)

        CCF, tshift = crosscorrel_on_NWB_quantity(Q1=Q, Q2=None,
                         q1=qq, t_q1=times, q2=dFoF, t_q2=data.Neuropil.timestamps[:], tmax=20)
        AX[2+i][3].plot(tshift, CCF, '-', color=plt.cm.tab10(i))
        AX[2+i][3].set_xlabel('time (s)', fontsize=10)
        AX[2+i][3].set_ylabel('cross correl.', fontsize=10)
        

    return fig


def behavior_analysis_fig(data,
                          running_speed_threshold=0.1):

    MODALITIES, QUANTITIES, TIMES, UNITS = find_modalities(data)
    n = len(MODALITIES)+(len(MODALITIES)-1)
        
    plt.style.use('ggplot')

    fig, AX = plt.subplots(n, 4, figsize=(11.4, 2.5*n))
    if n==1:
        AX=[AX]
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.3/n, top=0.95, wspace=.5, hspace=.6)

    for i, mod, quant, times, unit in zip(range(len(TIMES)), MODALITIES, QUANTITIES, TIMES, UNITS):
        color = plt.cm.tab10(i)
        AX[i][0].set_title(mod+40*' ', fontsize=10, color=color)
        quantity = (quant.data[:] if times is None else quant)
        AX[i][0].hist(quantity, bins=10,
                      weights=100*np.ones(len(quantity))/len(quantity), color=color)
        AX[i][0].set_xlabel(unit, fontsize=10)
        AX[i][0].set_ylabel('occurence (%)', fontsize=10)

        if mod=='Running-Speed':
            # do a small inset with fraction above threshold
            inset = AX[i][0].inset_axes([0.6, 0.6, 0.4, 0.4])
            frac_running = np.sum(quantity>running_speed_threshold)/len(quantity)
            inset.pie([100*frac_running, 100*(1-frac_running)], explode=(0, 0.1),
                      colors=[color, 'lightgrey'],
                      labels=['run ', ' rest'],
                      autopct='%.0f%%  ', shadow=True, startangle=90)
            inset.set_title('thresh=%.1fcm/s' % running_speed_threshold, fontsize=7)
            inset.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            
        quantity = (quant.data[:] if times is None else quant)
        AX[i][1].hist(quantity, bins=10,
                      weights=100*np.ones(len(quantity))/len(quantity), log=True, color=color)
        AX[i][1].set_xlabel(unit, fontsize=10)
        AX[i][1].set_ylabel('occurence (%)', fontsize=10)

        CC, ts = autocorrel_on_NWB_quantity(Q1=(quant if times is None else None),
                                            q1=(quantity if times is not None else None),
                                            t_q1=times,
                                            tmax=180)
        AX[i][2].plot(ts/60., CC, '-', color=color, lw=2)
        AX[i][2].set_xlabel('time (min)', fontsize=9)
        AX[i][2].set_ylabel('auto correl.', fontsize=9)
        
        CC, ts = autocorrel_on_NWB_quantity(Q1=(quant if times is None else None),
                                            q1=(quantity if times is not None else None),
                                            t_q1=times,
                                            tmax=20)
        AX[i][3].plot(ts, CC, '-', color=color, lw=2)
        AX[i][3].set_xlabel('time (s)', fontsize=9)
        AX[i][3].set_ylabel('auto correl.', fontsize=9)

    for i1 in range(len(UNITS)):
        m1, q1, times1, unit1 = MODALITIES[i1], QUANTITIES[i1], TIMES[i1], UNITS[i1]
        # for i2 in list(range(i1))+list(range(i1+1, len(UNITS))):
        for i2 in range(i1+1, len(UNITS)):
            
            i+=1
            m2, q2, times2, unit2 = MODALITIES[i2], QUANTITIES[i2], TIMES[i2], UNITS[i2]

            AX[i][0].set_title(m1+' vs '+m2+10*' ', fontsize=10)
            if times1 is None:
                Q1, qq1 = q1, None
            else:
                Q1, qq1 = None, q1
            if times2 is None:
                Q2, qq2 = q2, None
            else:
                Q2, qq2 = None, q2

            mean_q1, var_q1, mean_q2, var_q2 = crosshistogram_on_NWB_quantity(Q1=Q1, Q2=Q2,
                            q1=qq1, t_q1=times1, q2=qq2, t_q2=times2, Npoints=30)
        
            AX[i][0].errorbar(mean_q1, mean_q2, xerr=var_q1, yerr=var_q2, color='k')
            AX[i][0].set_xlabel(unit1, fontsize=10)
            AX[i][0].set_ylabel(unit2, fontsize=10)

            mean_q1, var_q1, mean_q2, var_q2 = crosshistogram_on_NWB_quantity(Q1=Q2, Q2=Q1,
                            q1=qq2, t_q1=times2, q2=qq1, t_q2=times1, Npoints=30)
        
            AX[i][1].errorbar(mean_q1, mean_q2, xerr=var_q1, yerr=var_q2, color='k')
            AX[i][1].set_xlabel(unit2, fontsize=10)
            AX[i][1].set_ylabel(unit1, fontsize=10)

            
            CCF, tshift = crosscorrel_on_NWB_quantity(Q1=Q2, Q2=Q1,
                            q1=qq2, t_q1=times2, q2=qq1, t_q2=times1,\
                                                      tmax=180)
            AX[i][2].plot(tshift/60, CCF, 'k-')
            AX[i][2].set_xlabel('time (min)', fontsize=10)
            AX[i][2].set_ylabel('cross correl.', fontsize=10)

            CCF, tshift = crosscorrel_on_NWB_quantity(Q1=Q2, Q2=Q1,
                                        q1=qq2, t_q1=times2, q2=qq1, t_q2=times1,\
                                        tmax=20)
            AX[i][3].plot(tshift, CCF, 'k-')
            AX[i][3].set_xlabel('time (s)', fontsize=10)
            AX[i][3].set_ylabel('cross correl.', fontsize=10)
            
    return fig


def add_inset_with_time_sample(TLIM, tlim):
    # inset with time sample
    axT = plt.axes([0.6, 0.9, 0.3, 0.05])
    axT.axis('off')
    axT.plot(tlim, [0,0], 'k-', lw=2)
    axT.plot(TLIM, [0,0], '-', color=plt.cm.tab10(3), lw=5)
    axT.annotate('0 ', (0,0), xycoords='data', ha='right', fontsize=9)
    axT.annotate(' %.1fmin' % (tlim[1]/60.), (tlim[1],0), xycoords='data', fontsize=9)

def make_summary_pdf(filename, Nmax=1000000,
                    include=['exp', 'rois', 'raw', 'protocols'],
                    T_raw_data=60,
                    N_raw_data=5,
                    ROI_raw_data=20,
                    verbose=True):

    data = MultimodalData(filename)
    
    if bool(data.metadata['CaImaging']):
        data.roiIndices = np.sort(np.random.choice(np.arange(data.iscell.sum()),
                                               size=min([data.iscell.sum(), ROI_raw_data]),
                                               replace=False))
    else:
        data.roiIndices = []

    folder = summary_pdf_folder(filename)
    
    if not os.path.isdir(folder):
        os.mkdir(folder)
        print('-> created summary PDF folder !')
    else:
        print('summary PDF folder already exists !')
        
    if 'exp' in include:
        with PdfPages(os.path.join(folder, 'exp.pdf')) as pdf:

            print('* writing experimental metadata as "exp.pdf" [...] ')
            
            print('   - notes')
            fig = metadata_fig(data)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            print('   - behavior analysis ')
            fig = behavior_analysis_fig(data)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

    if 'rois' in include:
        
        with PdfPages(os.path.join(folder, 'rois.pdf')) as pdf:

            print('* plotting ROI analysis as "rois.pdf" [...]')
            print('   - plotting imaging FOV')
            fig, AX = plt.subplots(1, 4, figsize=(11.4, 2.))
            data.show_CaImaging_FOV(key='meanImg', NL=1, cmap='viridis', ax=AX[0])
            data.show_CaImaging_FOV(key='meanImg', NL=2, cmap='viridis', ax=AX[1])
            data.show_CaImaging_FOV(key='meanImgE', NL=2, cmap='viridis', ax=AX[2])
            data.show_CaImaging_FOV(key='max_proj', NL=2, cmap='viridis', ax=AX[3])
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
            for i in data.roiIndices:
                print('   - plotting analysis of ROI #%i' % (i+1))
                fig = roi_analysis_fig(data, roiIndex=i)
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
                
        
    if 'raw' in include:
        
        with PdfPages(os.path.join(folder, 'raw.pdf')) as pdf:

            print('* plotting raw data as "raw.pdf" [...]')
            fig, ax = plt.subplots(1, figsize=(11.4, 5))
            fig.subplots_adjust(top=0.8, bottom=0.05)
            print('   - plotting full data view')
            data.plot_raw_data(data.tlim,
                               settings=raw_data_plot_settings(data, subsampling_factor=int(2*(data.tlim[1]-data.tlim[0])/60.+1)),
                               ax=ax, Tbar=int((data.tlim[1]-data.tlim[0])/30.))
            axT = add_inset_with_time_sample(data.tlim, data.tlim)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
            
            # plot raw data sample
            for t0 in np.linspace(T_raw_data, data.tlim[1], N_raw_data):
                TLIM = [np.max([10,t0-T_raw_data]),t0]
                print('   - plotting raw data sample at times ', TLIM)
                fig, ax = plt.subplots(1, figsize=(11.4, 5))
                fig.subplots_adjust(top=0.8, bottom=0.05)
                data.plot_raw_data(TLIM, settings=raw_data_plot_settings(data),
                                   ax=ax, Tbar=int(T_raw_data/30.))
                axT = add_inset_with_time_sample(TLIM, data.tlim)
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
                
        
    if 'protocols' in include:
        print('* looping over protocols for analysis [...]')
        
        # looping over protocols
        for p, protocol in enumerate(data.protocols):

            print('* * plotting protocol "%s" [...]' % protocol)
            
            with PdfPages(os.path.join(folder, '%s.pdf' % protocol)) as pdf:
            
                # finding protocol type
                protocol_type = (data.metadata['Protocol-%i-Stimulus' % (p+1)] if (len(data.protocols)>1) else data.metadata['Stimulus'])
                # print(protocol_type)
                # then protocol-dependent analysis

                if protocol_type=='full-field-grating':
                    Nresp, SIs = 0, []
                    for i in data.roiIndices[:Nmax]:
                        print('   - plotting analysis of ROI #%i' % (i+1))
                        fig, SI, responsive = orientation_direction_selectivity.OS_ROI_analysis(data, roiIndex=i, verbose=False)
                        pdf.savefig()  # saves the current figure into a pdf page
                        plt.close()
                        if responsive:
                            Nresp += 1
                            SIs.append(SI)
                    # summary figure for this protocol
                    fig, AX = orientation_direction_selectivity.summary_fig(Nresp, data.iscell.sum(), np.array(SIs))
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                elif protocol_type=='drifting-full-field-grating':
                    Nresp, SIs = 0, []
                    for i in data.roiIndices[:Nmax]:
                        print('   - plotting analysis of ROI #%i' % (i+1))
                        fig, SI, responsive = orientation_direction_selectivity.DS_ROI_analysis(data, roiIndex=i, verbose=False)
                        pdf.savefig()  # saves the current figure into a pdf page
                        plt.close()
                        if responsive:
                            Nresp += 1
                            SIs.append(SI)
                    fig, AX = orientation_direction_selectivity.summary_fig(Nresp, data.iscell.sum(), np.array(SIs),
                                                                            label='Direction Select. Index')
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                elif protocol_type in ['center-grating', 'drifting-center-grating']:
                    from analysis.surround_suppression import orientation_size_selectivity_analysis
                    from analysis.orientation_direction_selectivity import summary_fig
                    Nresp, SIs = 0, []
                    for i in range(data.iscell.sum())[:Nmax]:
                        fig, responsive = orientation_size_selectivity_analysis(data,
                                                                roiIndex=i, verbose=False)
                        pdf.savefig()  # saves the current figure into a pdf page
                        plt.close()
                        if responsive:
                            Nresp += 1
                            SIs.append(0) # TO BE FILLED
                    fig, AX = summary_fig(Nresp, data.iscell.sum(), np.array(SIs),
                                          label='none')
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                elif 'noise' in protocol_type:
                    from receptive_field_mapping import RF_analysis
                    Nresp, SIs = 0, []
                    for i in range(data.iscell.sum())[:Nmax]:
                        fig, SI, responsive = RF_analysis(data, roiIndex=i, verbose=False)
                        pdf.savefig()  # saves the current figure into a pdf page
                        plt.close()
                        if responsive:
                            Nresp += 1
                            SIs.append(0) # TO BE FILLED
                    fig, AX = summary_fig(Nresp, data.iscell.sum(), np.array(SIs),
                                          label='none')
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                elif 'spatial-location' in protocol_type:
                    from surround_suppression import orientation_size_selectivity_analysis
                    Nresp, SIs = 0, []
                    for i in range(data.iscell.sum())[:Nmax]:
                        fig, responsive = orientation_size_selectivity_analysis(data, roiIndex=i, verbose=False)
                        pdf.savefig()  # saves the current figure into a pdf page
                        plt.close()
                        if responsive:
                            Nresp += 1
                            SIs.append(0) # TO BE FILLED
                    fig, AX = summary_fig(Nresp, data.iscell.sum(), np.array(SIs),
                                          label='none')
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

    print('[ok] pdfs succesfully saved in "%s" !' % folder)


    
def summary_pdf_folder(filename):
    return filename.replace('.nwb', '')


if __name__=='__main__':
    
    import argparse, datetime

    parser=argparse.ArgumentParser()
    parser.add_argument('-df', "--datafile", type=str,
                        default='/home/yann/DATA/CaImaging/NDNFcre_GCamp6s/2021_07_01/2021_07_01-16-27-22.nwb')
    parser.add_argument('-o', "--ops", type=str, nargs='*',
                        default=['exp'])
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    
    args = parser.parse_args()

    # pdf_dir = os.path.join(os.path.dirname(filename), 'summary', os.path.basename(filename))

    # fig1 = metadata_fig(data)
    # fig2 = behavior_analysis_fig(data)
    # data = MultimodalData(args.datafile)
    # fig3 = roi_analysis_fig(data, roiIndex=0)
    # plt.show()
    
    make_summary_pdf(args.datafile,
                     include=args.ops,
                     # include=['exp', 'raw', 'rois', 'protocols'],
                     Nmax=args.Nmax,
                     verbose=args.verbose)

    







