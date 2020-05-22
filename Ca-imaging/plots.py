from datavyz import ge

def activity_plot(graph, Data,
                  Data_original=None, t=None,
                  tzoom=[0, np.inf],
                  colors = None,
                  ax=None, bar_fraction=0.7, Tbar=10, lw=0.5):
    
    if ax is None:
        _, ax = graph.figure(axes_extents=(4,4))

    if colors is None:
        colors = [graph.colors[i%10] for i, k in enumerate(Data)]
        
    keys = [key for key in Data]
    if t is None:
        t = np.arange(len(Data[keys[0]]))
    t_cond = (t>=tzoom[0]) & (t>=tzoom[0])

    for i, key in enumerate(Data):
        if Data_original is not None:
            norm_factor = 1./(np.max(Data_original[key][t_cond])-np.min(Data_original[key][t_cond]))
            baseline = np.min(Data_original[key][t_cond])
            norm_Data_original = (Data_original[key][t_cond]-baseline)*norm_factor
            ax.plot(t[t_cond], i+norm_Data_original, colors[i], lw=0.2, alpha=.3)
        else:
            norm_factor = 1./(np.max(Data[key][t_cond])-np.min(Data[key][t_cond]))
            baseline = np.min(Data[key][t_cond]) 
           
        norm_Data = norm_factor*(Data[key][t_cond]-baseline)

        ax.plot(t[t_cond], i+norm_Data, colors[i], lw=lw)
        graph.annotate(ax, key, (t[t_cond][-1], i+1), color=colors[i],
                    xycoords='data', ha='right', size='small', va='top')
        
        # scale for that cell
        ax.plot([0, 0], [i, i+bar_fraction], color=graph.default_color)
        if 100.*norm_factor<1:
            graph.annotate(ax, '%.1f%%' % (100.*norm_factor),
                    (0, i), rotation=90, xycoords='data', ha='right', size='small')
        else:
            graph.annotate(ax, '%i%%' % int(100.*norm_factor),
                    (0, i), rotation=90, xycoords='data', ha='right', size='small')
        
    ax.plot([0, Tbar], [i+1, i+1], color=graph.default_color)
    graph.annotate(ax, '%is' % Tbar, (0, i+1), xycoords='data', size='small')
    
    ax.axis('off')
