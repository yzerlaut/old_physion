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

def analysis_pdf(datafile, Nmax=1000000):

    data = MultimodalData(datafile)

    # find the protocol with the many-standards
    iprotocol = np.argwhere([('many-standards' in p) for p in data.protocols])[0]

    if len(iprotocol)>0:
        iprotocol_MS = iprotocol[0]

        Nresp, SIs = 0, []

        pdf_OS = PdfPages(os.path.join(summary_pdf_folder(datafile), '%s-orientation_selectivity.pdf' % data.protocols[iprotocol]))
        
        for roi in np.arange(data.iscell.sum())[:Nmax]:

            fig, SI, responsive, responsive_angles = ODS.OS_ROI_analysis(data, roiIndex=roi,
                                                                         iprotocol=iprotocol_MS, with_responsive_angles=True)
            pdf_OS.savefig()  # saves the current figure into a pdf page
            plt.close()

            if responsive:
                Nresp += 1
                SIs.append(SI)

        ODS.summary_fig(Nresp, data.iscell.sum(), SIs, label='Orient. Select. Index')
        pdf_OS.savefig()  # saves the current figure into a pdf page
        plt.close()
        
        print('[ok] mismatch negativity analysis saved in: "%s" ' % summary_pdf_folder(datafile))
            
    else:
        print('"Many Standards" protocol not found')
                    

        
        
    print(iprotocol)


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




