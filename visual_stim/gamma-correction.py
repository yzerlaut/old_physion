import numpy as np
from datavyz import ges as ge
from scipy.optimize import minimize

LAMBDA = {'blue':'470nm', 'green':'635nm', 'red':'830nm'}

# values in microwatts (microwatt precision)
calib =  {'center': {'green':[0, 1, 2, 4, 7, 11, 15, 20, 25, 29, 32, 38, 44, 51, 58, 65, 71, 76, 81, 87], 'red':[0, 1, 2, 3, 6, 9, 12, 16, 19, 23, 25, 30, 35, 40, 46, 51, 56, 60, 64, 69], 'blue':[0, 2, 3, 5, 8, 13, 18, 24, 29, 34, 39, 45, 53, 61, 70, 77, 84, 91, 97, 104]},
              'top-left': {'green':[0, 1, 2, 4, 6, 10, 14, 18, 22, 26, 29, 34, 40, 46, 52, 57, 63, 67, 72, 78], 'red':[0, 1, 2, 3, 5, 8, 11, 15, 18, 21, 23, 27, 31, 36, 41, 45, 49, 53, 57, 61], 'blue':[0, 1, 3, 5, 8, 12, 17, 22, 27, 31, 35, 41, 47, 54, 62, 68, 75, 80, 86, 92]}}


# green before
gb = [3, 5, 8, 13, 20, 27, 33, 39, 44, 48, 54, 62, 69, 76, 83, 89, 95, 101, 106, 111]
rb = gb
bb = gb
# green after
gb = [3, 3, 4, 5, 7, 11, 16, 22, 29, 35, 41, 47, 53, 62, 72, 81, 91, 100, 107, 115]

ba = [3, 5, 10, 16, 24, 31, 37, 42, 47, 52, 59, 66, 73, 79, 86, 92, 98, 103, 107, 112]


calib =  {'center': {'green':gb, 'red':rb, 'blue':bb}}

lum = np.linspace(0, 1, len(gb))

def func(lum, coefs):
    # return coefs[0]+coefs[1]*lum**coefs[2]
    return coefs[0]*lum**coefs[1]

fig, AX = ge.figure(axes=(3,1))

# for location in ['center', 'top-left']:
for location in ['center']:
    for i, color in enumerate(['blue', 'green', 'red']):
        
        array = calib[location][color]
        array/=np.max(array)
        
        def to_minimize(coefs):
            return np.sum(np.abs(array-func(lum, coefs))**2)

        residual = minimize(to_minimize, [1, 1],
                            bounds=[(0.5, 2), (0.1, 3.)])

        print('For %s and %s, gamma=' % (location, color), residual.x[1])
        
        # ge.title(AX[i], "a=%.2f, k=%.2f, $\gamma$=%.2f" % (residual.x[0], residual.x[1], residual.x[2]), color=getattr(ge, color), size='small')
        ge.title(AX[i], "k=%.2f, $\gamma$=%.2f" % (residual.x[0], residual.x[1]), color=getattr(ge, color), size='small')
        ge.scatter(lum, array, ax=AX[i], color=getattr(ge, color), label='data', ms=3)
        ge.plot(lum, func(lum, residual.x), ax=AX[i], lw=3, alpha=.5, color=getattr(ge, color), label='fit')
        ge.annotate(AX[i],'$\lambda$=%s' % LAMBDA[color], (0.5,.1), color=getattr(ge, color))
        ge.set_plot(AX[i], xlabel='(computer) luminosity', xticks=[0,0.5, 1], yticks=[0,0.5, 1], ylabel='measured I (norm.)')

# fig.savefig('../doc/gamma-correction.png')
ge.show()
