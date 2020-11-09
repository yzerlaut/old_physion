import numpy as np
import skvideo.io
import imageio
import os, pathlib
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter
from suite2p.io.binary import BinaryFile


def compress_FluorescenceMovie(binaryFile,
                               smoothing=0, # either 2, or (1, 4, 5)
                               subsampling=1,
                               extension='.npz', # '.avi', or '.mp4', ...
                               tool='numpy',
                               Nframe_per_file=1000,
                               max_file=int(1e6),
                               verbose=False):


    data = BinaryFile(Ly=Ly, Lx=Lx, read_filename=binaryFile)

    """
    # create directory if not existing
    directory = os.path.join(datafolder, 'FaceCamera-compressed')
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------
    # building custom compression functions depending on the tool

    if tool=='numpy':
        
        extension='.npz' # forcing the extension in that case
        def compress_func(X, filename):
            np.savez_compressed(filename+extension, X)
            
    elif tool=='skvideo':

        if extension not in ['.avi', '.mp4']:
            print('forcing "mp4" extension')
            extension = '.mp4' 
        def compress_func(X, filename):
            writer = skvideo.io.FFmpegWriter(filename+extension)
            for i in range(X.shape[0]):
                writer.writeFrame(X[i,:,:])
            writer.close()
            
    elif tool=='imageio':

        if extension not in ['.avi', '.mp4']:
            print('forcing "mp4" extension')
            extension = '.mp4' 
        def compress_func(X, filename):
            imageio.mimwrite(filename+extension, X)

    # ------------------------------------------------
    # Now looping over frames to build the compression

    FILES = sorted(os.listdir(os.path.join(datafolder, 'FaceCamera-imgs')))
    print(FILES)
    if len(FILES)>0:
        X, i0, i, file_count = [], 0, 0, 0
        for i, fn in enumerate(FILES):
            x = np.load(os.path.join(datafolder, 'FaceCamera-imgs', fn))
            if smoothing!=0:
                x = gaussian_filter(x, smoothing)
            X.append(x)

            if ((i+1)%Nframe_per_file)==0:
                # we save
                filename = 'imgs-%i-%i' % (i0, i)
                compress_func(np.array(X), os.path.join(directory, filename))
                # and we reinit
                X, i0 = [], i+1
                if verbose:
                    print('wrote: ', filename)

                file_count +=1

            if file_count>=max_file:
                break

        # saving the last frames
        if file_count<max_file:
            filename = 'imgs-%i-%i' % (i0, i)
            compress_func(np.array(X), os.path.join(directory, filename))
            if verbose:
                print('wrote: ', filename)

        # saving compression metadata 
        compression_metadata = {'tool':tool,
                                'subsampling':subsampling,
                                'smoothing':smoothing,
                                'extension':extension,
                                'Nframe_per_file':Nframe_per_file,
                                'max_file':max_file}
        np.save(os.path.join(directory, 'compression-metadata.npy'), compression_metadata)

    """
def compress_datafolder(args):

    compress_FaceCamera(args.datafolder,
                        extension=args.extension,
                        tool=args.tool,
                        smoothing=args.smoothing,
                        verbose=args.verbose,
                        Nframe_per_file=args.Nframe_per_file,
                        max_file=args.max_file)
    
                
if __name__=='__main__':

    fn = '/home/yann/DATA/2020_11_03/TSeries-28102020-231-00-031/'

    xml = bruker_xml_parser(get_files_with_extension(fn, extension='.xml')[0])
    Lx, Ly = xml['settings']['pixelsPerLine'], xml['settings']['linesPerFrame']
    

    compress_FluorescenceMovie(Ly, Lx, os.path.join(fn, 'suite2p', 'plane0', 'data.bin'))
    
    # import argparse

    # parser=argparse.ArgumentParser()
    # # compression type
    # parser.add_argument("--extension", default='.mp4')
    # parser.add_argument("--tool", default='imageio')
    # parser.add_argument("--smoothing", type=int, default=2)
    # # Face Data
    # parser.add_argument("--Nframe_per_file", type=int, default=1000)
    # parser.add_argument("--max_file", type=int, default=100000)
    # # 
    # parser.add_argument('-df', "--datafolder", default='./')
    # parser.add_argument('-ddf', "--day_folder", default='')
    # parser.add_argument("--debug", action="store_true")
    # parser.add_argument('-v', "--verbose", action="store_true")
    # args = parser.parse_args()

    # import time, sys, pathlib
    # sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    # from assembling.saving import check_datafolder, list_dayfolder
    
    # if args.day_folder!='':
    #     for df in list_dayfolder(args.day_folder):
    #         args.datafolder = os.path.join(args.day_folder, df)
    #         if os.path.isdir(os.path.join(args.datafolder, 'FaceCamera-imgs')):
    #             print('checking %s [...]' % args.datafolder)
    #             try:
    #                 check_datafolder(args.datafolder)
    #             except BaseException as e:
    #                 print(e)
    #             print('compressing %s [...]' % args.datafolder)
    #             compress_datafolder(args)
    # else:
    #     tstart = time.time()
    #     print('Running extension:', args.extension,  'tool:' , args.tool)
    #     compress_datafolder(args)
    #     print('Compressed to:')
    #     os.system('du -sh %s ' % os.path.join(args.datafolder, 'FaceCamera-compressed', 'imgs-0-%i' % (args.Nframe_per_file-1) + args.extension))
    #     os.system('du -sh %s ' % os.path.join(args.datafolder, 'FaceCamera-compressed'))
    #     print('Original:')
    #     os.system('du -sh %s ' % os.path.join(args.datafolder, 'FaceCamera-imgs'))
    #     print(time.time()-tstart, 'seconds')
    
