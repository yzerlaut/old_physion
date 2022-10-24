import sys, os, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
# datavyz submodule
try:
    from dataviz.datavyz.datavyz import graph_env_manuscript as ge
except ModuleNotFoundError:
    ge = None
if ge is None:
    try:
        from datavyz import graph_env_manuscript as ge
    except ModuleNotFoundError:
        print(' \n Need to install "datavyz" with `pip install git+https://github.com/yzerlaut/datavyz  \n ')
from dataviz import show_data 
from dataviz import tools
from analysis.tools import *

def raw_data_plot_settings(data,
                           subsampling_factor=1):
    settings = {}
    if 'Running-Speed' in data.nwbfile.acquisition:
        settings['Locomotion'] = dict(fig_fraction=2, subsampling=subsampling_factor)
    if 'Pupil' in data.nwbfile.processing:
        settings['Pupil'] = dict(fig_fraction=2, subsampling=subsampling_factor)
    if 'Pupil' in data.nwbfile.processing:
        settings['GazeMovement'] = dict(fig_fraction=1, subsampling=subsampling_factor)
    if 'FaceMotion' in data.nwbfile.processing:
        settings['FaceMotion'] = dict(fig_fraction=2, subsampling=subsampling_factor)
    if not (subsampling_factor>1):
        settings['VisualStim'] = dict(fig_fraction=0.1, color='black')
    return settings


def analysis_fig(data,
                 running_speed_threshold=0.1):

    MODALITIES, QUANTITIES, TIMES, UNITS, COLORS = find_modalities(data)
    n = len(MODALITIES)+int(len(MODALITIES)*(len(MODALITIES)-1)/2)

    plt.style.use('ggplot')

    fig, AX = plt.subplots(n, 5, figsize=(11.4, 2.5*n))
    if n==1:
        AX=[AX]
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.3/n, top=0.9, wspace=.5, hspace=.6)

    for i, mod, quant, times, unit, color in zip(range(len(TIMES)), MODALITIES, QUANTITIES, TIMES, UNITS, COLORS):
        AX[i][0].set_title(mod, fontsize=10, color=color)

        quantity = (quant.data[:] if times is None else quant)
        
        if mod=='GazeMovement':
            nsamples = data.nwbfile.processing['Pupil'].data_interfaces['cx'].num_samples
            subsampling = np.max([1, int(nsamples/1000)])
            cx = data.nwbfile.processing['Pupil'].data_interfaces['cx'].data[:][::subsampling]
            cy = data.nwbfile.processing['Pupil'].data_interfaces['cy'].data[:][::subsampling]
            ge.scatter(cx-np.mean(cx), cy-np.mean(cy), ax=AX[i][0], color=color, ms=1, lw=0.05)
            ge.set_plot(AX[i][0], xlabel='x (mm)', ylabel='y (mm)')
            AX[i][0].axis('equal')
        else:
            AX[i][0].axis('off')
            
        if mod=='Running-Speed':
            inset = AX[i][0].inset_axes([0.2,0.2,0.6,0.6])
            frac_running = np.sum(quantity>running_speed_threshold)/len(quantity)
            D = np.array([100*frac_running, 100*(1-frac_running)])
            ge.pie([100*frac_running, 100*(1-frac_running)],
                   ax=inset,
                   # pie_labels = ['%.1f%%\n' % (100*d/D.sum()) for d in D],
                   COLORS=[color, 'lightgrey'],
                   ext_labels=['run \n%.1f%% ' % (100*frac_running), ' rest \n%.1f%% ' % (100*(1-frac_running))])
            ge.annotate(AX[0][0], 'thresh=%.1fcm/s' % running_speed_threshold, (0.5, 0), ha='center', va='top')
            

        AX[i][1].hist(quantity, bins=10,
                      weights=100*np.ones(len(quantity))/len(quantity), color=color)
        AX[i][1].set_xlabel(unit, fontsize=10)
        AX[i][1].set_ylabel('occurence (%)', fontsize=10)

        quantity = (quant.data[:] if times is None else quant)
        AX[i][2].hist(quantity, bins=10,
                      weights=100*np.ones(len(quantity))/len(quantity), log=True, color=color)
        AX[i][2].set_xlabel(unit, fontsize=10)
        AX[i][2].set_ylabel('occurence (%)', fontsize=10)

        CC, ts = autocorrel_on_NWB_quantity(Q1=(quant if times is None else None),
                                            q1=(quantity if times is not None else None),
                                            t_q1=times,
                                            tmax=180)
        AX[i][3].plot(ts/60., CC, '-', color=color, lw=2)
        AX[i][3].set_xlabel('time (min)', fontsize=9)
        AX[i][3].set_ylabel('auto correl.', fontsize=9)
        
        CC, ts = autocorrel_on_NWB_quantity(Q1=(quant if times is None else None),
                                            q1=(quantity if times is not None else None),
                                            t_q1=times,
                                            tmax=20)
        AX[i][4].plot(ts, CC, '-', color=color, lw=2)
        AX[i][4].set_xlabel('time (s)', fontsize=9)
        AX[i][4].set_ylabel('auto correl.', fontsize=9)

    for i1 in range(len(UNITS)):
        
        m1, q1, times1, unit1 = MODALITIES[i1], QUANTITIES[i1], TIMES[i1], UNITS[i1]

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

            hist, be1, be2 = hist2D_on_NWB_quantity(Q1=Q1, Q2=Q2,
                        q1=qq1, t_q1=times1, q2=qq2, t_q2=times2, bins=40)

            hist = np.log(np.clip(hist, np.min(hist[hist>0]), np.max(hist)))
            ge.matrix(hist, x=be1, y=be2, colormap=plt.cm.binary,
                      ax=AX[i][0], aspect='auto')
            AX[i][0].grid(False)
            AX[i][0].set_xlabel(unit2, fontsize=8)
            AX[i][0].set_ylabel(unit1, fontsize=8)
            ge.annotate(AX[i][0], '  log distrib.', (0,1), va='top', size='x-small')
            
            mean_q1, var_q1, mean_q2, var_q2 = crosshistogram_on_NWB_quantity(Q1=Q1, Q2=Q2,
                            q1=qq1, t_q1=times1, q2=qq2, t_q2=times2, Npoints=30)
        
            AX[i][1].errorbar(mean_q1, mean_q2, xerr=var_q1, yerr=var_q2, color='k')
            AX[i][1].set_xlabel(unit1, fontsize=10)
            AX[i][1].set_ylabel(unit2, fontsize=10)

            mean_q1, var_q1, mean_q2, var_q2 = crosshistogram_on_NWB_quantity(Q1=Q2, Q2=Q1,
                            q1=qq2, t_q1=times2, q2=qq1, t_q2=times1, Npoints=30)
        
            AX[i][2].errorbar(mean_q1, mean_q2, xerr=var_q1, yerr=var_q2, color='k')
            AX[i][2].set_xlabel(unit2, fontsize=10)
            AX[i][2].set_ylabel(unit1, fontsize=10)

            
            CCF, tshift = crosscorrel_on_NWB_quantity(Q1=Q2, Q2=Q1,
                            q1=qq2, t_q1=times2, q2=qq1, t_q2=times1,\
                                                      tmax=180)
            AX[i][3].plot(tshift/60, CCF, 'k-')
            AX[i][3].set_xlabel('time (min)', fontsize=10)
            AX[i][3].set_ylabel('cross correl.', fontsize=10)

            CCF, tshift = crosscorrel_on_NWB_quantity(Q1=Q2, Q2=Q1,
                                        q1=qq2, t_q1=times2, q2=qq1, t_q2=times1,\
                                        tmax=20)
            AX[i][4].plot(tshift, CCF, 'k-')
            AX[i][4].set_xlabel('time (s)', fontsize=10)
            AX[i][4].set_ylabel('cross correl.', fontsize=10)
            
    return fig


def analysis_pdf(datafile,
                 Tzoom=120,
                 NzoomPlot=3):

    pdf_filename = os.path.join(summary_pdf_folder(datafile), 'behavior.pdf')

    data = show_data.MultimodalData(datafile)
    
    with PdfPages(pdf_filename) as pdf:

        print('* plotting behavioral data as "behavior.pdf" [...]')

        print('   - raw behavior plot ')
        fig, ax = plt.subplots(1, figsize=(11.4, 3.5))
        fig.subplots_adjust(top=0.8, bottom=0.05)
        
        subsampling = max([int((data.tlim[1]-data.tlim[0])/data.CaImaging_dt/1000), 1])
        data.plot_raw_data(data.tlim,
                           settings=raw_data_plot_settings(data, subsampling_factor=int((data.tlim[1]-data.tlim[0])/60.+1)),
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
            data.plot_raw_data(TLIM,
                               settings=raw_data_plot_settings(data,
                                                               subsampling_factor=1), ax=ax)
            axT = add_inset_with_time_sample(TLIM, data.tlim, plt)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

        
        print('   - behavior analysis ')
        fig = analysis_fig(data)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    
    print('[ok] behavioral data saved as: "%s" ' % pdf_filename)


def population_analysis(FILES,
                        min_time_minutes=2,
                        exclude_subjects=[],
                        ax=None,
                        running_speed_threshold=0.1):

    times, fracs_running, subjects = [], [], []
    if ax is None:
        fig, ax = ge.figure(figsize=(5,1), top=5)
    else:
        fig = None
        
    for f in FILES:

        data = show_data.MultimodalData(f)
        if (data.nwbfile is not None) and ('Running-Speed' in data.nwbfile.acquisition):
            speed = data.nwbfile.acquisition['Running-Speed'].data[:]
            max_time = len(speed)/data.nwbfile.acquisition['Running-Speed'].rate
            if (max_time>60*min_time_minutes) and (data.metadata['subject_ID'] not in exclude_subjects):
                times.append(max_time)
                fracs_running.append(100*np.sum(speed>running_speed_threshold)/len(speed))
                subjects.append(data.metadata['subject_ID'])
            
    i=-1
    for c, s in enumerate(np.unique(subjects)):
        s_cond = np.array(subjects)==s
        ax.bar(np.arange(1+i, i+1+np.sum(s_cond)),
               np.array(fracs_running)[s_cond]+1,
               width=.75, color=plt.cm.tab10(c%10))
        i+=np.sum(s_cond)+1
    ax.bar([i+2], [np.mean(fracs_running)], yerr=[np.std(fracs_running)],
           width=1.5, color='grey')
    ge.annotate(ax, 'frac. running:\n%.1f+/-%.1f %%' % (np.mean(fracs_running), np.std(fracs_running)),
                (i+3, np.mean(fracs_running)), xycoords='data')
    ge.set_plot(ax, xticks=[], xlabel='\nrecording', ylabel='       frac. running (%)')
    ymax, i = ax.get_ylim()[1], -1
    for c, s in enumerate(np.unique(subjects)):
        s_cond = np.array(subjects)==s
        ge.annotate(ax, s, (1+i, ymax), rotation=90, color=plt.cm.tab10(c%10), xycoords='data', size='x-small')
        i+=np.sum(s_cond)+1
    return fig, ax

    
if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument("--running_speed_threshold", help='in cm/s', type=float, default=0.1)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    
    if '.nwb' in args.datafile:
        analysis_pdf(args.datafile)
    elif os.path.isdir(args.datafile):
        from assembling.saving import get_files_with_extension
        FILES = get_files_with_extension(args.datafile, extension='.nwb', recursive=True)
        population_analysis(FILES,
                            exclude_subjects=['Mouse1_NDNF', 'Mouse1_WT', 'Mouse1_VIP-Cre', 'Mouse2_WT', 'Mouse4_WT', 'Mouse5_WT', 'Mouse1_CB1_TdTomato'])
    else:
        print('/!\ Need to provide a NWB datafile as argument ')
                               
    if args.debug:
        os.system('xdg-open %s' % os.path.join(summary_pdf_folder(args.datafile), 'behavior.pdf'))
                               
                               










