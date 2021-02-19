import os, sys
import numpy as np
from scipy.optimize import minimize
from analyz.signal_library.classical_functions import exp_thresh

def step(t, t0, t1):
    return (np.sign(t-t0)-np.sign(t-t1))/2.

def heaviside(t, t1):
    return (np.sign(t-t1)+1)/2.

def VCfunc_to_fit(t, coeffs, t0=50, t1=150):
    Ibsl, IbslShift, IexpComp, Tau = coeffs
    return Ibsl+\
        step(t, t0, t1)*(-IbslShift-IexpComp*exp_thresh(-(t-t0)/Tau))+\
        heaviside(t, t1)*IexpComp*exp_thresh(-(t-t1)/Tau)


def extract_VCcharact(t, data, t0=50, t1=300,
                      with_plot=False, ge=None, title=''):
    
    TforBaseline = [t0-(t1-t0)/3., t0] 
    TforBaselineShift = [t1-(t1-t0)/3., t1] 
    TforPeak = [t0, t0+(t1-t0)/3.] 
    Ibsl = data[(t>TforBaseline[0])&(t<TforBaseline[1])].mean()
    IbslShift = np.abs(data[(t>TforBaselineShift[0]) & (t<TforBaselineShift[1])].mean()-Ibsl)
    IexpPeak = np.max(np.abs(data[(t>TforPeak[0]) & (t<TforPeak[1])]-Ibsl))

    def to_minimize(coefs):
        return np.mean(np.abs(data-VCfunc_to_fit(t, coefs, t0=t0, t1=t1)))
    
    res = minimize(to_minimize,
                   [1,1,1,1], method='SLSQP', bounds=[(0,np.inf) for i in range(4)])
    Tau = res.x[3]

    if with_plot:
        if ge is None:
            from datavyz import gen as ge
        fig, ax = ge.plot(t, data, fig_args={'figsize':(1.5,1.5)}, label='VC-data')
        ax.plot([t[0], t[-1]], Ibsl*np.ones(2), ':', lw=1, label='$I_{bsl}$')
        ax.plot([t[0], t[-1]], (Ibsl-IbslShift)*np.ones(2), ':', lw=1, label='$I_{bsl}^{shift}$')
        ax.plot([t[0], t[-1]], (Ibsl-IexpPeak)*np.ones(2), ':', lw=1, label='$I_{exp}^{peak}$')
        ax.plot(t0*np.ones(2), ax.get_ylim(), ':', lw=1, label='$t_0$')
        ax.plot(t1*np.ones(2), ax.get_ylim(), ':', lw=1, label='$t_1$')
        ax.plot(t, VCfunc_to_fit(t, res.x, t0=t0, t1=t1), 'w--', lw=0.5, label='fit for $\\tau$')
        ge.set_plot(ax, xlabel='time', ylabel='Irec')
        ge.legend(ax, loc=(1.,0.))
        ge.title(ax, title)
        return fig, ax, IbslShift, IexpPeak, Tau
    else:
        return IbslShift, IexpPeak, Tau

def from_VCcharact_to_membrane_parameters(IbslShift, IexpPeak, Tau,
                                        Vstep=5e-3, verbose=False):
    Rs = np.abs(Vstep)/IexpPeak
    C = Rs*IbslShift/np.abs(Vstep)
    Rm = Rs*(1-C)/C
    Cm = Tau*(Rm+Rs)/Rm/Rs
    if verbose:
        print("""
        membrane and recording parameters:
        - Rm=%.1fMOhm
        - Cm=%.1fpF
        - Rs=%.1fMOhm
        """ % (1e-6*Rm, 1e12*Cm, 1e-6*Rs))
    return Rm, Cm, Rs


def ICfunc_to_fit(t, coeffs, t0=50, t1=150):
    Vbsl, VbslShift, Tau = coeffs
    return Vbsl+step(t, t0, t1)*VbslShift*(1-exp_thresh(-(t-t0)/Tau))+\
        heaviside(t, t1)*VbslShift*exp_thresh(-(t-t1)/Tau)


def perform_ICcharact(t, data, t0=50e-3, t1=300e-3, Istep=200e-12,
                      with_plot=False, ge=None, title='', ax=None, fig_args={}):
    
    TforBaseline = [t0-(t1-t0)/3., t0] 
    TforBaselineShift = [t1-(t1-t0)/3., t1] 
    Vbsl = data[(t>TforBaseline[0])&(t<TforBaseline[1])].mean()
    Vresp = data[(t>TforBaselineShift[0]) & (t<TforBaselineShift[1])].mean()

    def to_minimize(coefs):
        return np.mean(np.abs(data-ICfunc_to_fit(t, coefs, t0=t0, t1=t1)))
    
    # res = minimize(to_minimize, [-80e-3,-75e-3,20e-3])
    res = minimize(to_minimize, [-80e-3,-75e-3,20e-3],
                   method='SLSQP', bounds=[(-np.inf,np.inf),
                                           (-np.inf,np.inf),
                                           (1e-4,500e-3)])
    Tau = res.x[2]

    Rm = (Vresp-Vbsl)/Istep
    Cm = Tau/Rm
    
    if with_plot:
        if ge is None:
            from datavyz import gen as ge
        if ax is None:
            fig, ax = ge.plot(t, data, fig_args=fig_args, label='IC-data')
        else:
            fig=None
        ax.plot([t[0], t[-1]], Vbsl*np.ones(2), ':', lw=1, label='$V_{bsl}$')
        ax.plot([t[0], t[-1]], Vresp*np.ones(2), ':', lw=1, label='$V_{resp}$')
        ax.plot(t0*np.ones(2), ax.get_ylim(), ':', lw=1, label='$t_0$')
        ax.plot(t1*np.ones(2), ax.get_ylim(), ':', lw=1, label='$t_1$')
        ax.plot(t, ICfunc_to_fit(t, res.x, t0=t0, t1=t1), '--',
                label='fit for $\\tau$', color='k', lw=5, alpha=.5)
        ge.set_plot(ax, xlabel='time', ylabel='$V_m$')
        ge.legend(ax, loc=(1.,0.))
        ge.title(ax, title)
        ge.annotate(ax, """
        - Rm=%.1fMOhm
        - Cm=%.1fpF
        """ % (1e-6*Rm, 1e12*Cm), (.6,.5))
        return fig, ax, Rm, Cm
    else:
        return Rm, Cm

