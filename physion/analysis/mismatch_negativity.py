import sys, os, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from datavyz import graph_env_manuscript as ge

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz.show_data import MultimodalData
from analysis.tools import summary_pdf_folder
from analysis.process_NWB import EpisodeResponse
from analysis import orientation_direction_selectivity as ODS
from scipy.stats import ttest_rel

def modulation_summary_panel(t, data,
                             title ='',
                             pre_interval=[-1, 0],
                             post_interval=[1, 2],
                             responsive_only=False):

    if responsive_only:
        fig, AX = ge.figure(axes=(4,1), wspace=2)
        valid_cells = np.array(data['responsive'])
        # pie plot for responsiveness
        AX[0].pie([100*np.sum(valid_cells)/len(valid_cells),
                   100*(1-np.sum(valid_cells)/len(valid_cells))],
                  explode=(0, 0.1),
                  colors=[plt.cm.tab10(2), plt.cm.tab10(3)],
                  labels=['responsive', 'unresponsive'],
                  autopct='%1.1f%%', shadow=True, startangle=90)
        
        iax = 1
    else:
        fig, AX = ge.figure(axes=(3,1), wspace=2)
        iax=0
        valid_cells = np.ones(len(data['control']), dtype=bool)

    ge.annotate(fig, '%s, n=%i ROIs' % (title, np.sum(valid_cells)), (0.05, 0.98), va='top')

    pre_cond = (t>pre_interval[0]) & (t<pre_interval[1])
    post_cond = (t>post_interval[0]) & (t<post_interval[1])

    evoked_levels = {}
    if np.sum(valid_cells)>1:
        for ik, key, color in zip(range(3), ['redundant', 'control', 'deviant'],
                                  [ge.blue, 'k', ge.red]):
            y = np.array(data[key])[valid_cells,:]
            ge.plot(t, y.mean(axis=0), sy=y.std(axis=0), color=color, ax=AX[iax], no_set=True)
            y_bsl = y-np.mean(y[:,pre_cond], axis=1).reshape((np.sum(valid_cells),1))
            ge.plot(t,y_bsl.mean(axis=0),sy=y_bsl.std(axis=0),color=color,ax=AX[iax+1],no_set=True)
            # bar plot
            evoked_levels[key] = np.mean(y_bsl[:,post_cond], axis=1)
            AX[iax+2].bar([ik], np.mean(evoked_levels[key]),
                          yerr=np.std(evoked_levels[key]), lw=2, color=color)

        # adaptation effect (control vs redundant conditions)
        test = ttest_rel(evoked_levels['redundant'], evoked_levels['control'])
        AX[iax+2].plot([0,1], AX[iax+2].get_ylim()[1]*np.ones(2), 'k-', lw=1)
        ge.annotate(AX[iax+2], ge.from_pval_to_star(test.pvalue), (0.5,AX[iax+2].get_ylim()[1]),
                    ha='center', xycoords='data')

        # surprise effect (control vs deviant conditions)
        test = ttest_rel(evoked_levels['deviant'], evoked_levels['control'])
        AX[iax+2].plot([1,2], AX[iax+2].get_ylim()[1]*np.ones(2), 'k-', lw=1)
        ge.annotate(AX[iax+2], ge.from_pval_to_star(test.pvalue), (1.5,AX[iax+2].get_ylim()[1]),
                    ha='center', xycoords='data')

        # pre / post intervals
        ylim1 = AX[iax].get_ylim()[0]
        AX[iax].plot(pre_interval, ylim1*np.ones(2), lw=1, color='dimgrey')
        AX[iax].plot(post_interval, ylim1*np.ones(2), lw=1, color='dimgrey')
        
        ge.set_plot(AX[iax], xlabel='time (s)', ylabel='dF/F')
        ge.set_plot(AX[iax+1], xlabel='time (s)', ylabel='$\Delta$ dF/F')
        ge.title(AX[iax+1], 'baseline corrected', size='x-small')
        ge.set_plot(AX[iax+2], xlabel='', ylabel='evoked $\Delta$ dF/F')

    return fig, AX
        
    
def analysis_pdf(datafile, Nmax=1000000):

    data = MultimodalData(datafile)

    stim_duration = data.metadata['Protocol-1-presentation-duration']
    interval_post = [1./2.*stim_duration, stim_duration]
    interval_pre = [interval_post[0]-interval_post[1], 0]
    
    try:
        # find the protocol with the many-standards
        iprotocol_MS = np.argwhere([('many-standards' in p) for p in data.protocols])[0][0]
        # find the protocol with the oddball-1
        iprotocol_O1 = np.argwhere([('oddball-1' in p) for p in data.protocols])[0][0]
        # find the protocol with the oddball-2
        iprotocol_O2 = np.argwhere([('oddball-2' in p) for p in data.protocols])[0][0]

        # mismatch negativity angles
        MM_angles = [data.metadata['Protocol-%i-angle-redundant (deg)' % (1+iprotocol_O1)],
                     data.metadata['Protocol-%i-angle-deviant (deg)' % (1+iprotocol_O1)]]
        # 
        DATA = {'stim_duration':stim_duration}
        DATA[str(int(MM_angles[0]))] = {'iprotocol_control':iprotocol_MS, 'control':[], 'redundant':[], 'deviant':[], 'responsive':[]}
        DATA[str(int(MM_angles[1]))] = {'iprotocol_control':iprotocol_MS, 'control':[], 'redundant':[], 'deviant':[], 'responsive':[]}
        DATA[str(int(data.metadata['Protocol-%i-angle-redundant (deg)' % (1+iprotocol_O1)]))]['iprotocol_redundant'] = iprotocol_O1
        DATA[str(int(data.metadata['Protocol-%i-angle-deviant (deg)' % (1+iprotocol_O1)]))]['iprotocol_deviant'] = iprotocol_O1
        DATA[str(int(data.metadata['Protocol-%i-angle-redundant (deg)' % (1+iprotocol_O2)]))]['iprotocol_redundant']= iprotocol_O2
        DATA[str(int(data.metadata['Protocol-%i-angle-deviant (deg)' % (1+iprotocol_O2)]))]['iprotocol_deviant']=iprotocol_O2
        
        # find the angle for the redundant and deviant conditions
        print(data.metadata['Protocol-3-angle-redundant (deg)'], data.metadata['Protocol-3-angle-deviant (deg)'])
        
        Nresp, Nresp_selective, SIs = 0, 0, []

        pdf_OS = PdfPages(os.path.join(summary_pdf_folder(datafile), '%s-orientation_selectivity.pdf' % data.protocols[iprotocol_MS]))
        pdf_MSO = PdfPages(os.path.join(summary_pdf_folder(datafile), '%s-mismatch_selective_only.pdf' % data.protocols[iprotocol_MS]))
        pdf_MA = PdfPages(os.path.join(summary_pdf_folder(datafile), '%s-mismatch_all.pdf' % data.protocols[iprotocol_MS]))
        
        for roi in np.arange(data.iscell.sum())[:Nmax]:

            print('   - MMN analysis for ROI #%i / %i' % (roi+1, data.iscell.sum()))
            ## ORIENTATION SELECTIVITY ANALYSIS
            fig, SI, responsive, responsive_angles = ODS.OS_ROI_analysis(data, roiIndex=roi,
                                                                         iprotocol=iprotocol_MS,
                                                                         stat_test_props=dict(interval_pre=interval_pre,
                                                                                              interval_post=interval_post,
                                                                                              test='wilcoxon', positive=True),
                                                                         with_responsive_angles=True)
            pdf_OS.savefig()  # saves the current figure into a pdf page
            plt.close()

            if responsive:
                Nresp += 1
                SIs.append(SI)

            EPISODES = EpisodeResponse(data,
                                       protocol_id=None, # means all
                                       quantity='CaImaging', subquantity='dF/F',
                                       roiIndex = roi)

            
            fig, AX = ge.figure(axes=(2,1), wspace=3., right=10.)
            responsive_for_at_least_one = False
            ge.annotate(fig, 'ROI #%i' % (roi+1), (0.02, 0.98), va='top')
            
            for angle, ax in zip(MM_angles, AX):
                
                DATA[str(int(angle))]['responsive'].append(False) # False by default
                
                ge.title(ax, '$\\theta$=%.1f$^{o}$' % angle)
                for ik, key, color in zip(range(3), ['control', 'redundant', 'deviant'], ['k', ge.blue, ge.red]):
                    cond = data.get_stimulus_conditions([np.array(DATA[str(int(angle))]['iprotocol_%s' % key]), np.array([float(angle)])], ['protocol_id', 'angle'], None)[0]
                    ge.plot(EPISODES.t, EPISODES.resp[cond,:].mean(axis=0), sy=EPISODES.resp[cond,:].std(axis=0),
                            color=color, ax=ax, no_set=True)
                    ge.annotate(ax, ik*'\n'+'%s, n=%i' % (key, np.sum(cond)), (0.98, 1.), color=color, va='top', size='small')
                    
                    # storing for population analysis:
                    DATA[str(int(angle))][key].append(EPISODES.resp[cond,:].mean(axis=0))
                    if angle in responsive_angles:
                        responsive_for_at_least_one = True
                        DATA[str(int(angle))]['responsive'][-1] = True # shift to True
                        
                ge.set_plot(ax, xlabel='time (s)', ylabel='dF/F')

            pdf_MA.savefig()
            if responsive_for_at_least_one:
                pdf_MSO.savefig()
            plt.close()


        for angle in MM_angles:
            fig, AX = modulation_summary_panel(EPISODES.t, DATA[str(int(angle))],
                                               title='$\\theta$=%.1f$^{o}$' % angle)
            pdf_MA.savefig()
            plt.close()
            fig, AX = modulation_summary_panel(EPISODES.t, DATA[str(int(angle))],
                                               title='$\\theta$=%.1f$^{o}$' % angle,
                                               responsive_only=True)
            pdf_MSO.savefig()
            plt.close()
        
        # orientation selectivity summary
        ODS.summary_fig(Nresp, data.iscell.sum(), SIs, label='Orient. Select. Index')
        pdf_OS.savefig()  # saves the current figure into a pdf page
        plt.close()

        # modulation summary
    
        for pdf in [pdf_OS, pdf_MSO, pdf_MA]:
            pdf.close()
        
        print('[ok] mismatch negativity analysis saved in: "%s" ' % summary_pdf_folder(datafile))
            
    except BaseException as be:
        print('\n', be)
        print('---------------------------------------')
        print(' /!\ Pb with mismatch negativity analysis /!\  ')
        


if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    # parser.add_argument("analysis", type=str, help='should be either "orientation"/"direction"')
    # parser.add_argument("--iprotocol", type=int, default=0, help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if '.nwb' in args.datafile:
        analysis_pdf(args.datafile, Nmax=args.Nmax)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')




