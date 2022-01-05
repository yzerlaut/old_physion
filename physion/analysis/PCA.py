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
                 n_components=-1):
        """
        input has convention: (Nfeatures, Nsamples) , e.g. (Ncells, Ntimesamples)
        """

        # initialize quantity
        if (X is None) and (quantity=='CaImaging'):
            X = Ca_imaging_tools.compute_CaImaging_trace(data, subquantity,
                                                         range(data.iscell.sum()))
            self.t = data.Neuropil.timestamps[:]

        self.Nfeatures, self.Nsamples = X.shape # HERE !!
        
        self.means, self.stds = np.mean(X, axis=1), np.std(X, axis=1)
        
        if n_components<0:
            n_components = self.Nfeatures
        self.components = 1+np.arange(n_components)
            
        # initialize PCA from scikit-learn
        super().__init__(n_components=n_components)

        # rescale
        self.X = X.T/self.stds-self.means
        
        # perform PCA
        self.fit(self.X)

        # get the component projections
        self.Xinv = self.fit_transform(self.X)

    def get_projection(self, component_ID=0):
        return self.Xinv[:,component_ID]

    def projected_activity(self, component_ID=0):
        
        if type(component_ID) in [list, np.array, range, np.ndarray]:
            component_IDs = component_ID
        else:
            component_IDs = [component_ID]
        
        output = np.zeros((self.Nfeatures, self.Nsamples))
        for c, comp in enumerate(component_IDs):
            # print(comp)
            # print(self.get_projection(comp).shape)
            output += np.array([self.get_projection(comp)*self.components_[c][i] for i in range(self.Nfeatures)])
            
        return np.array([self.means[i]+self.stds[i]*output[i,:] for i in range(self.Nfeatures)])

    
    def show_explained_variance(self, xticks_subsampling=2, graph_env=ge):
        # let's plot the variance explained by the components
        fig, ax = graph_env.figure()
        graph_env.plot(self.components, 100.*self.explained_variance_ratio_,
                m='o', ms=3, ax=ax, no_set=True)
        graph_env.set_plot(ax, xlabel='component #', ylabel='% var. expl.',
                           xticks=np.arange(1, len(self.components))[::xticks_subsampling])
        return fig, ax
    
    def show_components(self, component_ID, ylim=None, graph_env=ge,
                        fig_args=dict(figsize=(1.4,.5), top=2, reshape_axes=False, hspace=0.7, left=0.8)):

        if type(component_ID) in [list, np.array, range]:
            component_IDs = component_ID
        else:
            component_IDs = [component_ID]

        fig, AX = graph_env.figure(axes=(1, len(component_IDs)), **fig_args)

        for i, ax in enumerate(graph_env.flat(AX)):
            ax.plot(self.components_[component_IDs[i]], 'k-', lw=1)
            ax.plot([0, self.Nfeatures-1], [0,0], 'k:', lw=0.5)
            
        for i, ax in enumerate(graph_env.flat(AX)):
            if i==0:
                graph_env.set_plot(ax, ['top', 'left'], xlabel='n= %i rois' % self.Nfeatures, ylabel='PC #%i' % (component_IDs[i]+1),
                            xminor_ticks=range(self.Nfeatures), xticks=[])
            else:
                graph_env.set_plot(ax, ['left'], ylabel='PC #%i' % (component_IDs[i]+1))
        graph_env.set_common_ylims(AX, lims=ylim)
        
        return fig, AX

    
if __name__=='__main__':

    fn = '/home/yann/DATA/CaImaging/NDNFcre_GCamp6s/Batch-2_September_2021/2021_09_10/2021_09_10-13-52-49.nwb'
    # fn = sys.argv[-1]

    from physion.dataviz.show_data import MultimodalData
    from physion.analysis.process_NWB import EpisodeResponse
    
    data = MultimodalData(fn)

    pca = PCA(data)

    # pca.show_explained_variance()

    # pca.show_components(0)
    # pca.show_components(range(1))

    # EPISODES = EpisodeResponse(data,
    #                            protocol_id=0,
    #                            quantity=pca.get_projection(0),
    #                            tfull=data.Neuropil.timestamps[:])

    # data.plot_trial_average(EPISODES=EPISODES, 
    #                         protocol_id=0,
    #                         color_keys=['contrast', 'speed'],
    #                         column_key='angle')


    raster = pca.projected_activity(np.arange(1))

    # normalize here for raster plot
    norm_raster = np.array([(raster[i,:]-np.min(data.Fluorescence.data[i,:]))/(np.max(data.Fluorescence.data[i,:])-np.min(data.Fluorescence.data[i,:])) for i in range(raster.shape[0])])
    
    tlim = [0,300]
    fig, ax = data.plot_raw_data(tlim,
                                 settings={'Locomotion':dict(fig_fraction=2, subsampling=30, color=ge.blue),
                                           'FaceMotion':dict(fig_fraction=2, subsampling=30, color=ge.purple),
                                           'Pupil':dict(fig_fraction=2, subsampling=10, color=ge.red),
                                           'CaImagingRaster':dict(fig_fraction=5, subsampling=1,
                                                                  roiIndices='all',
                                                                  normalization='per-line',
                                                                  quantity='CaImaging', subquantity='Fluorescence')},
                                 Tbar=10)
    ge.title(ax, 'Full data')
    
    fig, ax = data.plot_raw_data(tlim,
                                 settings={'Locomotion':dict(fig_fraction=2, subsampling=30, color=ge.blue),
                                           'FaceMotion':dict(fig_fraction=2, subsampling=30, color=ge.purple),
                                           'Pupil':dict(fig_fraction=2, subsampling=10, color=ge.red),
                                           'CaImagingRaster':dict(fig_fraction=5, subsampling=1, raster=norm_raster,
                                                                  roiIndices='all',
                                                                  normalization='None')},
                       Tbar=10)
    ge.title(ax, 'projection of PC components')


    fig, AX = ge.figure(axes=(1,10), figsize=(3,.2))
    cond = (pca.t>tlim[0]) & (pca.t>tlim[1])

    for i in range(10):
        AX[i].plot(raster[i,cond], color=ge.blue, lw=1)
        AX[i].plot(data.Fluorescence.data[i,:][cond], color='k', lw=0.5)

    for ax in AX:
        ge.set_plot(ax, ['left'])
    ge.show()
    






