import sys, os, pathlib
import numpy as np

from datavyz import graph_env_manuscript as ge

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from sklearn.decomposition import PCA as sklPCA
from sklearn import preprocessing
from Ca_imaging import tools as Ca_imaging_tools


class PCA(sklPCA):
    """
    adding a few handy tools to integrate the PCA of sklearn to the nwbfile format
    
    """
    def __init__(self, data, X=None,
                 quantity='CaImaging', subquantity='Fluorescence',
                 n_components=1):


        super().__init__(n_components=n_components)

        self.components = 1+np.arange(n_components)
        
        if (X is None) and (quantity=='CaImaging'):
            X = Ca_imaging_tools.compute_CaImaging_trace(data, subquantity,
                                                         range(data.iscell.sum())).T
            self.t = data.Neuropil.timestamps[:]
            self.Nsamples = len(self.t)
            self.Nfeatures = data.iscell.sum()

        # scale signals
        self.scaler = preprocessing.StandardScaler().fit(X)
        self.X = self.scaler.transform(X)

        # perform PCA
        self.fit(self.X)

        # get the component projections
        self.Xinv = self.fit_transform(self.X)

    def get_projection(self, component_ID=0):
        return self.Xinv[:,component_ID]
        
    def show_explained_variance(self, xticks_subsampling=2):
        # let's plot the variance explained by the components
        fig, ax = ge.figure()
        ge.plot(self.components, 100.*self.explained_variance_ratio_,
                m='o', ms=3, ax=ax, no_set=True)
        ge.set_plot(ax, xlabel='component #', ylabel='% var. expl.', xticks=np.arange(1, len(self.components))[::xticks_subsampling])
        return fig, ax
    
    def show_components(self):
        pass
        

    
if __name__=='__main__':

    fn = '/home/yann/DATA/CaImaging/NDNFcre_GCamp6s/Batch-2_September_2021/2021_09_10/2021_09_10-13-52-49.nwb'
    # fn = sys.argv[-1]

    from physion.analysis.read_NWB import Data
    data = Data(fn)

    pca = PCA(data, n_components=10)

    pca.show_explained_variance()

    ge.plot(pca.t[::100], pca.get_projection()[::100])
    ge.show()
    





