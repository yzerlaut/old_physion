import sys, time, tempfile, os, pathlib, json, datetime, string
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz.show_data import MultimodalData
from analysis.tools import *

def raw_data_plot_settings(data):
    settings = {}
    if 'Photodiode-Signal' in data.nwbfile.acquisition:
        settings['Photodiode'] = dict(fig_fraction=0.1, subsampling=100, color='grey')
    if 'Running-Speed' in data.nwbfile.acquisition:
        settings['Locomotion'] = dict(fig_fraction=1, subsampling=100, color='b')
    if 'Pupil' in data.nwbfile.processing:
        settings['Pupil'] = dict(fig_fraction=2, subsampling=2, color='red')
    if 'Whisking' in data.nwbfile.processing:
        settings['Whisking'] = dict(fig_fraction=2, subsampling=2, color='red')
    if 'ophys' in data.nwbfile.processing:
        settings['CaImaging'] = dict(fig_fraction=8, roiIndices=data.roiIndices, quantity='CaImaging',
                                     subquantity='dF/F', vicinity_factor=1., color='tab', subsampling=10)
        # settings['CaImagingSum'] = dict(fig_fraction=2,
        #                                 subsampling=10, 
        #                                 quantity='CaImaging', subquantity='dF/F', color='green')
    settings['VisualStim'] = dict(fig_fraction=0.01, color='black')
    return settings


def exp_analysis_fig(data):
    
    plt.style.use('ggplot')
    fig, AX = plt.subplots(1, 4, figsize=(11.4, 2))
    plt.subplots_adjust(left=0.01, right=0.97, bottom=0.3, wspace=1.)
    print(data.nwbfile.stimulus)
    s=''
    for key in ['protocol', 'subject_ID', 'notes']:
        s+='- %s :\n    "%s" \n' % (key, data.metadata[key])
    s += '- completed:\n       n=%i/%i episodes' %(data.nwbfile.stimulus['time_start_realigned'].data.shape[0], data.nwbfile.stimulus['time_start'].data.shape[0])
    AX[0].annotate(s, (0,1), va='top', fontsize=9)
    AX[0].axis('off')

    if 'Running-Speed' in data.nwbfile.acquisition:
        speed = np.abs(data.nwbfile.acquisition['Running-Speed'].data[:])
        AX[1].hist(speed, weights=100*np.ones(len(speed))/len(speed))
        AX[1].set_xlabel('speed', fontsize=10)
        AX[1].set_ylabel('occurence (%)', fontsize=10)
    else:
        AX[1].axis('off')

    if 'Pupil' in data.nwbfile.acquisition:
        diameter=data.nwbfile.processing['Pupil'].data_interfaces['sx'].data[:]*data.nwbfile.processing['Pupil'].data_interfaces['sy'].data[:]
        AX[2].hist(diameter, log=True)
        AX[2].set_xlabel('Pupil diam.', fontsize=10)
        AX[2].set_ylabel('count', fontsize=10)
    else:
        AX[2].annotate('no pupil', (.5,.5), va='center', ha='center', fontsize=10)
        AX[2].axis('off')

    return fig


def roi_analysis_fig(data, roiIndex=0):
    
    plt.style.use('ggplot')
    fig, AX = plt.subplots(1, 4, figsize=(11.4, 4))
    plt.subplots_adjust(left=0.01, right=0.97, bottom=0.3, wspace=.5)
    AX[0].annotate(' ROI#%i' % (roiIndex+1), (0.03, 0.1), xycoords='figure fraction', weight='bold', fontsize=11)

    data.show_CaImaging_FOV(key='meanImgE', cmap='viridis', ax=AX[0], roiIndex=roiIndex)
    data.show_CaImaging_FOV(key='meanImgE', cmap='viridis', ax=AX[1], roiIndex=roiIndex, with_roi_zoom=True)

    index = np.arange(len(data.iscell))[data.iscell][roiIndex]
    if 'Running-Speed' in data.nwbfile.acquisition:
        CCF, tshift = crosscorrel_on_NWB_quantity(\
                            Q1=data.nwbfile.acquisition['Running-Speed'],
                            t_q2=data.Fluorescence.timestamps[:],
                            q2=data.Fluorescence.data[index,:],
                            tmax=20)
        AX[2].plot(tshift, CCF)
        AX[2].set_xlabel('time (s)', fontsize=10)
        AX[2].set_ylabel('correl.', fontsize=10)
        # AX[2].set_ylim([-0.3,1.])
        AX[2].set_title('running speed', fontsize=10)
        if (np.max(CCF)<0.2) and (np.min(CCF)>-0.1):
            AX[2].set_ylim([-0.1,0.2])
            AX[2].set_yticks([-0.1,0, 0.1, 0.2])

    if 'Pupil' in data.nwbfile.acquisition:
        CCF, tshift = crosscorrel_on_NWB_quantity(t_q1=data.nwbfile.processing['Pupil'].data_interfaces['sx'].timestamps[:],
                                                  q1=data.nwbfile.processing['Pupil'].data_interfaces['sx'].data[:]*data.nwbfile.processing['Pupil'].data_interfaces['sy'].data[:],
                            t_q2=data.Fluorescence.timestamps[:],
                            q2=data.Fluorescence.data[data.iscell[roiIndex],:],
                                                  tmax=100)
        AX[3].plot(tshift, CCF)
        AX[3].set_xlabel('time (s)', fontsize=10)
        AX[3].set_ylabel('correl.', fontsize=10)
        AX[3].set_title('pupil', fontsize=10)
        

    return fig


def behavior_analysis_fig(data):

    MODALITIES, QUANTITIES, TIMES, UNITS = [], [], [], []
    if 'Running-Speed' in data.nwbfile.acquisition:
        MODALITIES.append('Running-Speed')
        QUANTITIES.append(data.nwbfile.acquisition['Running-Speed'])
        TIMES.append(None)
        UNITS.append('cm/s')
    if 'Pupil' in data.nwbfile.processing:
        MODALITIES.append('Pupil')
        diameter=data.nwbfile.processing['Pupil'].data_interfaces['sx'].data[:]*\
            data.nwbfile.processing['Pupil'].data_interfaces['sy'].data[:]
        QUANTITIES.append(diameter)
        TIMES.append(data.nwbfile.processing['Pupil'].data_interfaces['sy'].timestamps[:])
        UNITS.append('mm$^2$')
    if 'Whisking' in data.nwbfile.processing:
        MODALITIES.append('Whisking')
        
    plt.style.use('ggplot')
    fig, AX = plt.subplots(len(MODALITIES)+(len(MODALITIES)-2)*(len(MODALITIES)-1), 4,
                 figsize=(11.4, 2.5*len(MODALITIES)+(len(MODALITIES)-2)*(len(MODALITIES)-1)))
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.3,
                        wspace=.5, hspace=.6)


    for i, mod, quant, times, unit in zip(range(len(TIMES)), MODALITIES, QUANTITIES, TIMES, UNITS):
        color = plt.cm.tab10(i)
        AX[i][0].set_title(mod, fontsize=10, color=color)
        quantity = (quant.data[:] if times is None else quant)
        AX[i][0].hist(quantity, weights=100*np.ones(len(quantity))/len(quantity), color=color)
        AX[i][0].set_xlabel(unit, fontsize=10)
        AX[i][0].set_ylabel('occurence (%)', fontsize=10)

        quantity = (quant.data[:] if times is None else quant)
        AX[i][1].hist(quantity, weights=100*np.ones(len(quantity))/len(quantity), log=True, color=color)
        AX[i][1].set_xlabel(unit, fontsize=10)
        AX[i][1].set_ylabel('occurence (%)', fontsize=10)

        print(quantity)
        CC, ts = autocorrel_on_NWB_quantity(Q1=(quant if times is None else None),
                                            q1=(quant.data[:] if times is not None else None),
                                            t_q1=times,
                                            tmax=120)
        AX[i][2].plot(ts, CC, '-', color=color)
        AX[i][2].set_xlabel('time (s)', fontsize=9)
        AX[i][2].set_ylabel('correl.', fontsize=9)
        
        CC, ts = autocorrel_on_NWB_quantity(Q1=(quant if times is None else None),
                                            q1=(quant.data[:] if times is not None else None),
                                            t_q1=times,
                                            tmax=20)
        AX[i][3].plot(ts, CC, '-', color=color)
        AX[i][3].set_xlabel('time (s)', fontsize=9)
        AX[i][3].set_ylabel('correl.', fontsize=9)
        
        # axins = AX[1].inset_axes((1-0.5,1-0.5,.5,.5))
        # CC, ts = autocorrel_on_NWB_quantity(q1=diameter,
        #     t_q1=data.nwbfile.processing['Pupil'].data_interfaces['sx'].timestamps[:],
        #                                     tmax=20)
        # axins.plot(ts, CC, 'k-')
        # axins.set_ylim([0,1.05])
        
    #     CC, ts = autocorrel_on_NWB_quantity(Q1=data.nwbfile.acquisition['Running-Speed'],
    #                                         tmax=120)
    #     AX[0].plot(ts, CC, 'k-')
        
    #     axins = AX[0].inset_axes((1-0.5,1-0.5,.5,.5))
    #     CC, ts = autocorrel_on_NWB_quantity(Q1=data.nwbfile.acquisition['Running-Speed'],
    #                                         tmax=10)
    #     axins.plot(ts, CC, 'k-')
    #     axins.set_ylim([0,1.05])
        
    #     AX[0].set_xlabel('time (s)', fontsize=9)
    #     AX[0].set_ylabel('correl.', fontsize=9)
        
    #     if 'Whisking' in data.nwbfile.processing:
    #         pass
    #     else:
    #         AX[4].axis('off')

    # if 'Pupil' in data.nwbfile.processing:
        
    #     diameter=data.nwbfile.processing['Pupil'].data_interfaces['sx'].data[:]*\
    #         data.nwbfile.processing['Pupil'].data_interfaces['sy'].data[:]

        
    #     if 'Running-Speed' in data.nwbfile.acquisition:
    #         mean_q1, var_q1, mean_q2, var_q2 = crosshistogram_on_NWB_quantity(\
    #             Q2=data.nwbfile.acquisition['Running-Speed'],
    #             t_q1=data.nwbfile.processing['Pupil'].data_interfaces['sx'].timestamps[:],
    #             q1=diameter,
    #             Npoints=30)
        
    #         AX[3].errorbar(mean_q1, mean_q2, xerr=var_q1, yerr=var_q2, color='k')
    #         AX[3].set_xlabel('pupil diam.', fontsize=10)
    #         AX[3].set_ylabel('run. speed (cm/s)', fontsize=10)
    #     else:
    #         AX[3].axis('off')
    # else:
    #     AX[3].axis('off')
    #     AX[1].axis('off')

    # if 'Whisking' in data.nwbfile.processing:
    #     pass
    # else:
    #     AX[2].axis('off')
            
    return fig


def make_sumary_pdf(filename, Nmax=1000000,
                    T_raw_data=180, N_raw_data=3, ROI_raw_data=20, Tbar_raw_data=5):

    data = MultimodalData(filename)
    data.roiIndices = np.sort(np.random.choice(np.arange(data.iscell.sum()),
                                               size=min([data.iscell.sum(), ROI_raw_data]),
                                               replace=False))

    with PdfPages(filename.replace('nwb', 'pdf')) as pdf:

        # summary quantities for experiment
        print('plotting exp macro analysis ')
        fig = exp_analysis_fig(data)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
        
        fig = behavior_analysis_fig(data)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # plot imaging field of view
        fig, AX = plt.subplots(1, 4, figsize=(11.4, 2))
        print('plotting imaging FOV ')
        data.show_CaImaging_FOV(key='meanImg', NL=1, cmap='viridis', ax=AX[0])
        data.show_CaImaging_FOV(key='meanImg', NL=2, cmap='viridis', ax=AX[1])
        data.show_CaImaging_FOV(key='meanImgE', NL=2, cmap='viridis', ax=AX[2])
        data.show_CaImaging_FOV(key='max_proj', NL=2, cmap='viridis', ax=AX[3])
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
        
        # plot raw data sample
        for t0 in np.linspace(T_raw_data, data.tlim[1], N_raw_data):
            TLIM = [np.max([10,t0-T_raw_data]),t0]
            print('plotting raw data sample at times ', TLIM)
            fig, ax = plt.subplots(1, figsize=(11.4, 5))
            fig.subplots_adjust(top=0.8, bottom=0.05)
            data.plot(TLIM, settings=raw_data_plot_settings(data),
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
            print(protocol_type)
            # then protocol-dependent analysis
            
            if protocol_type=='full-field-grating':
                from analysis.orientation_direction_selectivity import orientation_selectivity_analysis
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
                from analysis.orientation_direction_selectivity import direction_selectivity_analysis
                Nresp, SIs = 0, []
                for i in range(data.iscell.sum())[:Nmax]:
                    fig, SI, responsive = direction_selectivity_analysis(data, roiIndex=i, verbose=False)
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()
                    if responsive:
                        Nresp += 1
                        SIs.append(SI)
                fig, AX = summary_fig(Nresp, data.iscell.sum(), np.array(SIs),
                                      label='Direction Select. Index')
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
            
            elif protocol_type in ['center-grating']:
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

            elif 'noise' in protocol_type:
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


                
                
        print('[ok] pdf succesfully saved as "%s" !' % filename.replace('nwb', 'pdf'))        


def summary_fig(Nresp, Ntot, quantity,
                label='Orient. Select. Index',
                labels=['responsive', 'unresponsive']):
    fig, AX = plt.subplots(1, 4, figsize=(11.4, 2.5))
    fig.subplots_adjust(left=0.1, right=0.8, bottom=0.2)
    AX[1].pie([100*Nresp/Ntot, 100*(1-Nresp/Ntot)], explode=(0, 0.1),
              colors=[plt.cm.tab10(2), plt.cm.tab10(3)],
              labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    AX[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    AX[3].hist(quantity)
    AX[3].set_xlabel(label, fontsize=9)
    AX[3].set_ylabel('count', fontsize=9)
    for ax in [AX[0], AX[2]]:
        ax.axis('off')
    return fig, AX
    

    
if __name__=='__main__':
    
    # filename = '/home/yann/DATA/Wild_Type/2021_03_11-17-13-03.nwb'
    filename = sys.argv[-1]
    data = MultimodalData(filename)
    # fig1 = exp_analysis_fig(data)
    fig2 = behavior_analysis_fig(data)
    # fig3 = roi_analysis_fig(data, roiIndex=7)
    plt.show()
    
    # make_sumary_pdf(filename)
    
