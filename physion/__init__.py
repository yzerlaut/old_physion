from . import analysis
try: 
    from .dataviz.datavyz.datavyz import graph_env
    from . import dataviz, visual_stim, assembling, intrinsic
except BaseException as be:
    print(' loaded reduced version of "physion" without submodules' )

