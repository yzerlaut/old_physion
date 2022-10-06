import os, sys, pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from physion.intrinsic import Analysis
from physion.dataviz.datavyz.datavyz import graph_env
ge = graph_env('screen') # for display on screen

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser(description="Experiment interface",
                       formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("datafolder", type=str,default='')
    parser.add_argument('-v', "--verbose", action="store_true")
    
    args = parser.parse_args()
    maps = Analysis.load_maps(args.datafolder)

    Analysis.plot_retinotopic_maps(maps, 'altitude')
    Analysis.plot_retinotopic_maps(maps, 'azimuth')

    ge.show()

