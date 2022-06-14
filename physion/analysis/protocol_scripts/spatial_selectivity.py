import sys, os, pathlib
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from datavyz import graph_env_manuscript as ge

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from analysis.read_NWB import Data
from dataviz.show_data import MultimodalData, EpisodeResponse, format_key_value
from analysis.tools import summary_pdf_folder
from analysis import stat_tools


def summary_fig(results):

    other_keys = []
    for key in results:
        if (key not in ['Ntot', 'significant', 'x-center', 'y-center', 'value']) and ('-bins' not in key) and ('relative_' not in key):
            other_keys.append(key)

    fig, AX = ge.figure(axes=(2+len(other_keys),1), right=2, figsize=(1., 1.1))

    if ('x-center' in results) and ('y-center' in results):
        ge.hist2d(results['x-center'], results['y-center'],
                  bins=(results['x-center-bins'], results['y-center-bins']),
                  ax=AX[0],
                  xlabel='x-center ($^{o}$)', ylabel='y-center ($^{o}$)',
                  title='max. resp.')
    elif ('x-center' in results):
        ge.hist(results['x-center'], bins=results['x-center-bins'], ax=AX[0], xlabel='x-center ($^{o}$)')
    elif ('y-center' in results):
        ge.hist(results['y-center'], bins=results['y-center-bins'], ax=AX[0], xlabel='y-center ($^{o}$)')
    else:
        AX[0].axis('off')
    
    for i, key in enumerate(other_keys):
        ge.hist(results[key], bins=results[key+'-bins'], ax=AX[i+1], xlabel=key, title='max resp')
        AX[i+1].set_xlabel(key, fontsize=8)
        
    ge.annotate(AX[-1], 'n=%i/%i resp. cells' % (np.sum(results['significant']), results['Ntot']), (0.5,.0), ha='center', va='top')

    frac_resp = np.sum(results['significant'])/results['Ntot']
    data = np.array([100*frac_resp, 100*(1-frac_resp)])
    ge.pie(data,
           COLORS=[plt.cm.tab10(2), plt.cm.tab10(3)],
           pie_labels = ['  %.1f%%' % (100*d/data.sum()) for d in data],
           ext_labels = ['', ''],
           ax=AX[-1])
    
    return fig


def analysis_pdf(datafile,
                 iprotocol=0, 
                 stat_test_props=dict(interval_pre=[-1,0], interval_post=[0.5,1.5],
                                      test='anova', positive=True),
                 response_significance_threshold=0.05,
                 quantity='dFoF',
                 verbose=True,
                 Nmax=1000000):

    data = Data(datafile, metadata_only=True)
    
    print('   - computing episodes [...]')
    EPISODES = EpisodeResponse(datafile,
                               protocol_id=iprotocol,
                               quantities=['dFoF'])

    pdf_filename = os.path.join(summary_pdf_folder(datafile), '%s-spatial_selectivity.pdf' % data.protocols[iprotocol])
    
    results = {'Ntot':EPISODES.data.nROIs, 'significant':[]}

    fig, AX = ge.figure(axes=(3,1))
    
    with PdfPages(pdf_filename) as pdf:

        print('   - spatial-selectivity analysis for summed ROI fluo (n=%i rois)' % EPISODES.data.nROIs)
        
        fig, AX = EPISODES.plot_trial_average(quantity='dFoF', 
                                              roiIndices='all', roiIndex=None, 
                                              column_key='x-center', row_key='y-center', # color_key='angle',
                                              ybar=2, ybarlabel='2 s.d.',
                                              norm='Zscore-time-variations-after-trial-averaging-per-roi',
                                              with_std_over_rois=False,
                                              # ybar=0.2, ybarlabel='0.2dF/F',
                                              xbar=1., xbarlabel='1s',
                                              with_annotation=True,
                                              # with_std=False,
                                              with_stat_test=True, stat_test_props=stat_test_props,
                                              with_screen_inset=True,
                                              verbose=verbose)
        pdf.savefig(fig); plt.close(fig) # Add figure to pdf and close
        
        for roi in np.arange(EPISODES.data.nROIs)[:Nmax]:

            print('   - spatial-selectivity analysis for ROI #%i / %i' % (roi+1, EPISODES.data.nROIs))
            
            resp = EPISODES.compute_summary_data(stat_test_props,
                                                 exclude_keys=['repeat'],
                                                 response_args=dict(roiIndex=roi),
                                                 response_significance_threshold=response_significance_threshold)

            fig, AX = EPISODES.plot_trial_average(quantity=quantity, roiIndex=roi, 
                                                  column_key='x-center', row_key='y-center', color_key='angle',
                                                  ybar=0.2, ybarlabel='0.2dF/F',
                                                  xbar=1., xbarlabel='1s',
                                                  with_annotation=True,
                                                  with_std=True,
                                                  with_stat_test=True, stat_test_props=stat_test_props,
                                                  with_screen_inset=True,
                                                  verbose=verbose)
            pdf.savefig(fig); plt.close(fig) # Add figure to pdf and close

            significant_cond, label = (resp['significant']==True), '  -> max resp.: '
            if np.sum(significant_cond)>0:
                imax = np.argmax(resp['value'][significant_cond])
                for key in resp:
                    if ('-bins' not in key):
                        if (key not in results):
                            results[key] = [] # initialize if not done
                        results[key].append(resp[key][significant_cond][imax])
                        label+=format_key_value(key, resp[key][significant_cond][imax])+', ' # should have a unique value
                        print(label)
                ge.annotate(fig, label+'\n\n', (0,0), size='x-small')#, ha='right', va='top'), rotation=90)
            else:
                ge.annotate(fig, label+' N/A (no significant resp.)\n\n', (0, 0), size='x-small')#, ha='right', va='top', rotation=90)

        # the adding the bins
        for key in resp:
            if ('-bins' in key):
                results[key] = resp[key]
                
        fig = summary_fig(results)
        pdf.savefig(fig)  # saves the current figure into a pdf page
        plt.close(fig)

    print('[ok] spatial-selectivity analysis saved as: "%s" ' % pdf_filename)


if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument("--iprotocol", type=int, default=0, help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument('-st', "--significance_threshold", type=float, default=0.05)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if '.nwb' in args.datafile:
        analysis_pdf(args.datafile,
                     iprotocol=args.iprotocol,
                     Nmax=args.Nmax,
                     response_significance_threshold=args.significance_threshold)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')
        





