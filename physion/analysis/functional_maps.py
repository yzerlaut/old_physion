import os, sys, pathlib
import numpy as np
from PIL import Image
from matplotlib.cm import cool

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from physion.intrinsic import Analysis, RetinotopicMapping
from physion.dataviz.datavyz.datavyz import graph_env

ge = graph_env('manuscript') # for display on screen

def metadata_fig(datafolder):

    metadata = dict(np.load(os.path.join(datafolder, 'metadata.npy'), allow_pickle=True).item())

    metadata['recording-time'] = datafolder.split(os.path.sep)[-2:]
    if 'subject_props' in metadata:
        metadata['angle'] = metadata['subject_props']['headplate_angle_from_rig_axis_for_recording'] 
    else:
        metadata['angle'] = '...' 
    
    fig, ax = ge.figure(figsize=(4,1), left=0)

    string = """
    Mouse ID: "%(subject)s"

    Recorded @ %(recording-time)s

    headplate angle from rig/experimenter axis: %(angle)s
    """ % metadata
    ge.annotate(ax, string, (0,0), size='small')
    ax.axis('off')
    return fig, ax


def show_raw_data(t, data, params, maps,
                  pixel=(200,200)):
    
    fig, AX = ge.figure(axes_extents=[[[5,1]],[[5,1]],[[1,1] for i in range(5)]],
                        wspace=2.5, hspace=2.,
                        figsize=(0.7,0.6), left=1.5, top=1.5, bottom=1)


    AX[0][0].plot(t, data[:,pixel[0], pixel[1]], 'k', lw=1)
    ge.set_plot(AX[0][0], ylabel='pixel\n intensity (a.u.)', xlabel='time (s)',
                xlim=[t[0], t[-1]])
    # ge.annotate(AX[0][0], 'pixel: %s ' % pixel, (1,1),
                # ha='right', color='r', size='x-small')

    AX[1][0].plot(params['STIM']['up-times'], params['STIM']['up-angle'], 'k', lw=1)
    ge.set_plot(AX[1][0], ['left'], 
                ylabel='bar stim.\n angle ($^o$)',
                xlim=[t[0], t[-1]])

    ge.image(np.rot90(maps['vasculature'], k=1), ax=AX[2][0],
             title='green light')

    AX[2][1].scatter([pixel[0]], [pixel[1]], s=50, color='none', edgecolor='r', lw=1)
    ge.image(np.rot90(data[0,:,:], k=1), ax=AX[2][1],
             title='t=%.1fs' % t[0])

    AX[2][2].scatter([pixel[0]], [pixel[1]], s=50, color='none', edgecolor='r', lw=1)
    ge.image(np.rot90(data[-1,:,:], k=1), ax=AX[2][2],
             title='t=%.1fs' % t[-1])

    spectrum = np.fft.fft(data[:,pixel[0], pixel[1]], axis=0)
    
    power, phase = np.abs(spectrum), np.angle(spectrum)

    AX[2][3].plot(np.arange(1, len(power)), power[1:], color=ge.gray, lw=1)
    AX[2][3].plot([params['Nrepeat']], [power[params['Nrepeat']]], 'o', color=ge.blue, ms=4)
    ge.annotate(AX[2][3], 'stim. freq.', (0,0.01), va='top', size='small', color=ge.blue)

    AX[2][4].plot(np.arange(1, len(power)), phase[1:], color=ge.gray, lw=1)
    AX[2][4].plot([params['Nrepeat']], [phase[params['Nrepeat']]], 'o', color=ge.blue, ms=4)

    ge.set_plot(AX[2][3], ['left', 'top'], xscale='log', yscale='log', xlabelpad=4,
                xlim=[.99,101], ylim=[power[1:].max()/120.,1.5*power[1:].max()],
                xlabel='freq (sample unit)', ylabel='power (a.u.)')

    ge.set_plot(AX[2][4], ['left', 'top'], xscale='log', xlabelpad=3, 
                xlim=[.99,101], xlabel='freq.', ylabel='phase (Rd)')
    
    return fig


def build_pdf(args):

    width, height = int(8.27 * 300), int(11.7 * 300) # A4 at 300dpi
    page = Image.new('RGB', (width, height), 'white')

    fig_metadata, ax = metadata_fig(args.datafolder)
    fig_metadata.savefig('/tmp/fig_metadata.png', dpi=300)
    fig = Image.open('/tmp/fig_metadata.png')
    page.paste(fig, box=(200, 160))
    fig.close()

    maps = Analysis.load_maps(args.datafolder)
    maps['vasculature'] = (maps['vasculature']-np.min(maps['vasculature']))/(np.max(maps['vasculature'])-np.min(maps['vasculature']))
    maps['vasculature'] = maps['vasculature']**args.vasc_exponent

    # vasculature image
    fig, [ax,ax2] = ge.figure(axes=(2,1), figsize=(1.5,2.5), 
                              left=0, right=0, top=0.5, bottom=0, wspace=0.1)
    ax.imshow(maps['vasculature'], cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    ge.title(ax, 'green light', size='small')

    params, (t, data) = Analysis.load_raw_data(args.datafolder, 'up')

    # intrinsic imaging
    ax2.imshow(data[0,:,:], cmap='gray')
    ax2.axis('off')
    ge.title(ax2, 'red light, -500$\mu$m focus', size='small')

    fig.savefig('/tmp/fig.png', dpi=300)
    fig = Image.open('/tmp/fig.png')
    page.paste(fig, box=(int(3.5*300), int(150)))
    fig.close()

    fig_alt = Analysis.plot_retinotopic_maps(maps, 'altitude', ge=ge)
    fig_alt.savefig('/tmp/fig_alt.png', dpi=300)

    fig_azi = Analysis.plot_retinotopic_maps(maps, 'azimuth', ge=ge)
    fig_azi.savefig('/tmp/fig_azi.png', dpi=300)

    start, space = int(0.6*300), int(5.1*300)
    for name in ['alt', 'azi']:
        fig = Image.open('/tmp/fig_%s.png'%name)
        page.paste(fig, box=(start, space))
        # start+= fig.getbbox()[3]-fig.getbbox()[1] + 10
        start+= fig.getbbox()[2]-fig.getbbox()[0]
        fig.close()

    fig = show_raw_data(t, data, params, maps, pixel=args.pixel)
    fig.suptitle('example protocol: "up" ', fontsize=8)
    fig.savefig('/tmp/fig.png', dpi=300)
    fig = Image.open('/tmp/fig.png')
    page.paste(fig, box=(250, int(2.8*300)))
    fig.close()

    fig, AX = ge.figure(axes=(3,1), figsize=(1.5,2.5), 
                        left=0, right=0, top=1, bottom=0, wspace=0.2)
    AX[0].imshow(maps['vasculature'], cmap='gray', vmin=0, vmax=1)
    mean_power = maps['up-power']+maps['down-power']+maps['right-power']+maps['left-power']
    AX[0].imshow(mean_power, cmap=cool, alpha=0.3)
    AX[0].axis('off')
    ge.bar_legend(AX[0], 
            colorbar_inset={'rect':[0.1,1.05,0.8,0.07]},
            colormap=cool,
            label='mean power @ stim. freq.',
            orientation='horizontal')

    ge.bar_legend(AX[1], 
            colorbar_inset={'rect':[0.1,1.05,0.8,0.07]},
            colormap=ge.jet,
            label='sign of retinotopic gradient',
            orientation='horizontal')
    
    AX[1].imshow(maps['vasculature'], cmap='gray', vmin=0, vmax=1)
    AX[1].axis('off')

    trial_data = np.load(os.path.join(args.datafolder, 'analysis.npy'), allow_pickle=True).item()
    # trial_data = Analysis.build_trial_data(maps)
    trial = RetinotopicMapping.RetinotopicMappingTrial(**trial_data)
    trial.processTrial(isPlot=False)
    AX[1].imshow(trial.signMapf, cmap=ge.jet, alpha=0.7)


    AX[2].imshow(maps['vasculature'], cmap='gray', vmin=0, vmax=1)
    AX[2].axis('off')
    ge.title(AX[2], 'area segmentation')

    fig.savefig('/tmp/fig.png', dpi=300)
    fig = Image.open('/tmp/fig.png')
    page.paste(fig, box=(int(1*300), int(8.8*300)))
    fig.close()

    page.save('fig.pdf')


if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser(description="Experiment interface",
                       formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("datafolder", type=str,default='')
    parser.add_argument("--vasc_exponent", type=float,default=0.25)
    parser.add_argument("--pixel", type=int, nargs=2, default=(150,150))
    parser.add_argument('-v', "--verbose", action="store_true")
    
    args = parser.parse_args()

    build_pdf(args)
