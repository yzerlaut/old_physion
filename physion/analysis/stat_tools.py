import os, sys
import numpy as np
from scipy import stats

def stat_test_for_evoked_responses(EPISODES, episode_cond,
                                   interval_pre=[-2,0],
                                   interval_post=[1,3],
                                   test='wilcoxon'):


    pre_cond = (EPISODES['t']>=interval_pre[0]) & (EPISODES['t']<=interval_pre[1])
    post_cond = (EPISODES['t']>=interval_post[0]) & (EPISODES['t']<=interval_post[1])
    
    # print(EPISODES['resp'][episode_cond,post_cond])#.mean(axis=1))
    
    if test=='wilcoxon':
        return stats.wilcoxon(EPISODES['resp'][episode_cond,:][:,pre_cond].mean(axis=1),
                              EPISODES['resp'][episode_cond,:][:,post_cond].mean(axis=1))
    else:
        print(' "%s" test not implemented ! ' % test)

def pval_to_star(test, pvalue=1e-5, size=5):

    print(test)
    
    if test.pvalue<1e-3:
        return '***', size+1
    elif test.pvalue<1e-2:
        return '**', size+1
    elif test.pvalue<0.05:
        return '*', size+1
    else:
        return 'n.s.', size

        
if __name__=='__main__':

    # filename = sys.argv[-1]
    
    FullData= Data(filename)
    print(FullData.protocols)
    print('the datafile has %i validated ROIs (over %i from the full suite2p output) ' % (np.sum(FullData.iscell),
                                                                                          len(FullData.iscell)))
    # # for i in [2, 6, 9, 10, 13, 15, 16, 17, 21, 38, 41, 136]: # for 2021_03_11-17-13-03.nwb
    for i in range(np.sum(FullData.iscell))[:1]:
        fig, responsive = orientation_size_selectivity_analysis(FullData, roiIndex=i)
        if responsive:
            print('cell %i -> responsive !' % (i+1))
        plt.show()








    
