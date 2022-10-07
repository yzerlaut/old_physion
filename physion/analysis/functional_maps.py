import os, sys, pathlib
import numpy as np
from PIL import Image

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from physion.intrinsic import Analysis
from physion.dataviz.datavyz.datavyz import graph_env

ge = graph_env('manuscript') # for display on screen

def metadata_fig(datafolder):

    metadata = dict(np.load(os.path.join(datafolder, 'metadata.npy'), allow_pickle=True).item())

    metadata['recording-time'] = datafolder.split(os.path.sep)[-2:]

    fig, ax = ge.figure(figsize=(4,1), left=0)

    string = """
    Mouse ID: "%(subject)s"

    Recorded @ %(recording-time)s
    """ % metadata
    ge.annotate(ax, string, (0,0), size='small')
    ax.axis('off')
    return fig, ax


def build_pdf(datafolder):

    width, height = int(8.27 * 300), int(11.7 * 300) # A4 at 300dpi
    page = Image.new('RGB', (width, height), 'white')

    fig_metadata, ax = metadata_fig(datafolder)
    fig_metadata.savefig('/tmp/fig_metadata.png', dpi=300)
    fig = Image.open('/tmp/fig_metadata.png')
    page.paste(fig, box=(300, 200))

    """
    maps = Analysis.load_maps(args.datafolder)

    fig_alt = Analysis.plot_retinotopic_maps(maps, 'altitude', ge=ge)
    fig_alt.savefig('/tmp/fig_alt.png', dpi=300)

    fig_azi = Analysis.plot_retinotopic_maps(maps, 'azimuth', ge=ge)
    fig_azi.savefig('/tmp/fig_azi.png', dpi=300)

    start, space = int(0.3*300), int(5*300)
    for name in ['alt', 'azi']:
        fig = Image.open('/tmp/fig_%s.png'%name)
        page.paste(fig, box=(start, space))
        # start+= fig.getbbox()[3]-fig.getbbox()[1] + 10
        start+= fig.getbbox()[2]-fig.getbbox()[0]
        fig.close()

    """
    page.save('fig.pdf')


if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser(description="Experiment interface",
                       formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("datafolder", type=str,default='')
    parser.add_argument('-v', "--verbose", action="store_true")
    
    args = parser.parse_args()

    build_pdf(args.datafolder)
