import sys, pathlib
import numpy as np
import matplotlib.pylab as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from analysis.tools import *


def analysis_fig(data,
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



if __name__=='__main__':

    from analysis.read_NWB import Data
    
    if '.nwb' in sys.argv[-1]:
        data = Data(sys.argv[-1])
        fig = analysis_fig(data)
        plt.show()
    else:
        print('/!\ Need to provide a NWB datafile as argument ')








