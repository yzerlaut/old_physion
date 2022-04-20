"""
Script to remove the labels of some visual stim episodes so that they are not considered in the visually-evoked activity
we do this by simply clamping the photodiode signal to the baseline value, those episodes will thus disappear after "assembling"

"""
import os, tempfile, json, pathlib, shutil
import numpy as np
from matplotlib.pylab import plt

base_path = str(pathlib.Path(__file__).resolve().parents[1])

def update_NIdaq_data(args,
                      window=20,
                      subsampling=100):

    metadata = np.load(os.path.join(args.datafolder, 'metadata.npy'),
                       allow_pickle=True).item()
    
    if os.path.isfile(os.path.join(args.datafolder, 'NIdaq.start.npy')):
        NIdaq_Tstart = np.load(os.path.join(args.datafolder, 'NIdaq.start.npy'))[0]

        NIdaq_Tstart = np.load(os.path.join(args.datafolder, 'NIdaq.start.npy'))[0]
        

    data = np.load(os.path.join(args.datafolder, 'NIdaq.npy'), allow_pickle=True).item()
    t = np.arange(len(data['analog'][0,:]))/float(metadata['NIdaq-acquisition-frequency'])
    
    # checking that the two realisations are the same:
    fig, ax = plt.subplots(1, figsize=(7,5))
    cond = (t>(args.time-window)) & (t<(args.time+window))
    ax.plot(t[cond][::subsampling], data['analog'][0,cond][::subsampling], label='data')
    ax.plot(args.time*np.ones(2), ax.get_ylim(), 'r-', label='reset point')
    plt.legend()
    plt.show()


    y = input('  /!\ ----------------------- /!\ \n  Confirm that you want to remove episodes after the reset point ? [yes/No]\n')
    
    if y in ['y', 'Y', 'yes', 'Yes']:
        
        temp = str(tempfile.NamedTemporaryFile().name)+'.npy'
        print("""
        ---> moving the old data to the temporary file directory as: "%s" [...]
        """ % temp)
        shutil.move(os.path.join(args.datafolder, 'NIdaq.npy'), temp)
        print('applying changes [...]')
        cond = (t>args.time)
        data['analog'][0,cond] = data['analog'][0,cond][0]
        np.save(os.path.join(args.datafolder, 'NIdaq.npy'), data)
        print('done !')

    else:
        print('--> data update aborted !! ')
        



if __name__=='__main__':

    import argparse
    parser=argparse.ArgumentParser(description="""
    Script to add/modify the data in case of mistakes
    """,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('datafolder', type=str, help='session to clean')
    parser.add_argument('time', type=float, help='time after which we remove episodes')
    parser.add_argument('-v', "--verbose", action="store_true")
    args = parser.parse_args()

    update_NIdaq_data(args)




