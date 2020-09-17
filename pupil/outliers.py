import numpy as np
from scipy.interpolate import interp1d

def replace_outliers(data, std_criteria=2.):
    """
    Nearest-neighbor interpolation of pupil properties
    """
    product = np.ones(len(data['times']))
    std = 1
    for key in ['cx', 'cy', 'sx', 'sy', 'residual']:
        product *= np.abs(data[key]-data[key].mean())
        std *= std_criteria*data[key].std()
    accept_cond =  (product<std)
    
    dt = data['times'][1]-data['times'][0]
    for key in ['cx', 'cy', 'sx', 'sy', 'residual']:
        # duplicating the first and last valid points to avoid boundary errors
        x = np.concatenate([[data['times'][0]-dt], data['times'][accept_cond], [data['times'][-1]+dt]])
        y = np.concatenate([[data[key][accept_cond][0]],
                            data[key][accept_cond], [data[key][accept_cond][-1]]])
        func = interp1d(x, y, kind='nearest', assume_sorted=True)
        data[key+'-corrected'] = func(data['times'])
        
    return data

if __name__=='__main__':

    import sys
    if len(sys.argv)>1:
        fn = sys.argv[1]
    else:
        fn = '/home/yann/DATA/2020_09_11/13-40-10/pupil-data.npy'

    from datavyz import ges as ge

    data = np.load(fn, allow_pickle=True).item()

    fig, AX = ge.figure(axes=(1,6), figsize=(3,1), hspace=0.1)

    for i, key in enumerate(['cx', 'cy', 'sx', 'sy', 'residual']):
        AX[i].plot(data['times'], data[key], label='fit')
        
    data = replace_outliers(data)

    for i, key in enumerate(['cx', 'cy', 'sx', 'sy', 'residual']):
        AX[i].plot(data['times'], data[key+'-corrected'], label='corrected')
        ge.set_plot(AX[i], xlim=[data['times'][0], data['times'][-1]], ylabel=key)
        
    ge.show()

