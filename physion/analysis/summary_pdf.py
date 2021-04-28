import sys, time, tempfile, os, pathlib, json, datetime, string
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz.show_data import MultimodalData

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
        AX[1].hist(speed, log=True)
        AX[1].set_xlabel('speed', fontsize=10)
        AX[1].set_ylabel('count', fontsize=10)
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

    if 'Whisking' in data.nwbfile.acquisition:
        diameter=data.nwbfile.processing['Pupil'].data_interfaces['sx'].data[:]*data.nwbfile.processing['Pupil'].data_interfaces['sy'].data[:]
        AX[3].hist(diameter, log=True)
        AX[3].set_xlabel('Whisking signal (a.u.)', fontsize=10)
        AX[3].set_ylabel('count', fontsize=10)
    else:
        AX[3].annotate('no whisking', (.5,.5), va='center', ha='center', fontsize=10)
        AX[3].axis('off')
        
        # NEED TO RESAMPLE Pupil and Speed on the same times to do correlation
        t_pupil=data.nwbfile.processing['Pupil'].data_interfaces['sx'].timestamps[:]
        t_speed = np.arange(data.nwbfile.acquisition['Running-Speed'].num_samples)/data.nwbfile.acquisition['Running-Speed'].rate+data.nwbfile.acquisition['Running-Speed'].starting_time
        print(t_pupil[0], t_speed[0])
        print(t_pupil[-1], t_speed[-1])
        func=interp1d(t_pupil, diameter)
        
        new_diameter = func(t_speed)
        BINS=np.linspace(diameter.min(), diameter.max(), 20)
        bins = np.digitize(new_diameter, bins=BINS)
        for i, b in enumerate(np.unique(bins)):
            cond = (bins==b)
            AX[3].errorbar([new_diameter[cond].mean()],[speed[cond].mean()],
                           yerr=[speed[cond].std()],
                           xerr=[new_diameter[cond].std()])
        AX[3].axis('off')

    return fig



def behavior_analysis_fig(data):
    
    plt.style.use('ggplot')
    fig, AX = plt.subplots(1, 3, figsize=(11.4, 2))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, wspace=1.)
    
    if 'Running-Speed' in data.nwbfile.acquisition:
        speed = np.abs(data.nwbfile.acquisition['Running-Speed'].data[:])

    if 'Pupil' in data.nwbfile.acquisition:
        diameter=data.nwbfile.processing['Pupil'].data_interfaces['sx'].data[:]*data.nwbfile.processing['Pupil'].data_interfaces['sy'].data[:]

        # NEED TO RESAMPLE Pupil and Speed on the same times to do correlation
        t_pupil=data.nwbfile.processing['Pupil'].data_interfaces['sx'].timestamps[:]
        t_speed = np.arange(data.nwbfile.acquisition['Running-Speed'].num_samples)/data.nwbfile.acquisition['Running-Speed'].rate+data.nwbfile.acquisition['Running-Speed'].starting_time
        func=interp1d(t_pupil, diameter)
        
        new_diameter = func(t_speed)
        BINS=np.linspace(diameter.min(), diameter.max(), 20)
        bins = np.digitize(new_diameter, bins=BINS)
        for i, b in enumerate(np.unique(bins)):
            cond = (bins==b)
            AX[0].errorbar([new_diameter[cond].mean()],[speed[cond].mean()],
                           yerr=[speed[cond].std()],
                           xerr=[new_diameter[cond].std()], color='k')
        AX[0].set_xlabel('Pupil diameter', fontsize=10)
        AX[0].set_ylabel('Running speed', fontsize=10)
        
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
        if 'Pupil'
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
    fig1 = exp_analysis_fig(data)
    fig2 = behavior_analysis_fig(data)
    plt.show()
    
    # make_sumary_pdf(filename)
    
