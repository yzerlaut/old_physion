"""
Script to add/modify the data in case of mistakes
e.g. photodiode not connected

/!\ should be use only for the protocols that involve
    minimal calculations (e.g. gratings, ...) and not for the protocols
    where the calculations can induce jitters that can alter the time-course
    of the experiment from session to session !!

    
"""
import os, tempfile, json, pathlib, shutil
import numpy as np
from matplotlib.pylab import plt

base_path = str(pathlib.Path(__file__).resolve().parents[1])

def update_data(args, show_seconds=20, subsampling=10):

    # load previous
    metadata1 = np.load(os.path.join(args.session1, 'metadata.npy'), allow_pickle=True).item()
    data1 = np.load(os.path.join(args.session1, 'NIdaq.npy'), allow_pickle=True).item()
    t1 = np.arange(len(data1['analog'][0,:]))/metadata1['NIdaq-acquisition-frequency']
    
    metadata2 = np.load(os.path.join(args.session1, 'metadata.npy'), allow_pickle=True).item()
    data2 = np.load(os.path.join(args.session2, 'NIdaq.npy'), allow_pickle=True).item()
    t2 = np.arange(len(data2['analog'][0,:]))/metadata1['NIdaq-acquisition-frequency']


    metadata = np.load(os.path.join(args.session1, 'metadata.npy'), allow_pickle=True).item()
    data = np.load(os.path.join(args.session, 'NIdaq.npy'), allow_pickle=True).item()
    t = np.arange(len(data['analog'][0,:]))/metadata1['NIdaq-acquisition-frequency']

    tstart = np.min([t.max(), t1.max(), t2.max()])-show_seconds # showing the last 5 seconds


    # checking that the two realisations are the same:
    plt.figure(figsize=(7,5))

    plt.plot(t1[t1>tstart][::subsampling],
             data1[args.type][args.channel,t1>tstart][::subsampling], label='session #1')
    
    plt.plot(t2[t2>tstart][::subsampling],
             data2[args.type][args.channel,t2>tstart][::subsampling], label='session #2')

    plt.plot(t[t>tstart][::subsampling],
             data[args.type][args.channel,t>tstart][::subsampling], label='session')
    
    plt.legend()
    plt.show()


    y = input('  /!\ ----------------------- /!\ \n  Confirm that you want to replace\n the "%s" data of channel "%s" in session:\n "%s"\n by the data of session #1 ? [yes/No]' % (args.type, args.channel, args.session))
    if y in ['y', 'Y', 'yes', 'Yes']:
        
        temp = str(tempfile.NamedTemporaryFile().name)+'.npy'
        print("""
        ---> moving the old data to the temporary file directory as: "%s" [...]
        """ % temp)
        shutil.move(os.path.join(args.session, 'NIdaq.npy'), temp)
        print('applying changes [...]')
        data[args.type][args.channel,:] = data1[args.type][args.channel,:]
        np.save(os.path.join(args.session, 'NIdaq.npy'), data)
        print('done !')

    else:
        print('--> data update aborted !! ')
        



if __name__=='__main__':

    import argparse
    parser=argparse.ArgumentParser(description="""
    Script to add/modify the data in case of mistakes
    """,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('session1', type=str, help='example session #1')
    parser.add_argument('session2', type=str, help='example session #1')
    
    parser.add_argument('session', type=str, help='session to update !')

    parser.add_argument('-t', "--type", type=str, default='analog',
                        help='either "analog" or "digital" ')
    parser.add_argument('-c', "--channel", type=int, default=0,
                        help='provide the full path !')

    # parser.add_argument('-sf', "--subject_file", type=str,
    #     default=os.path.join(base_path, 'exp', 'subjects', 'mice_fani.json'))
    # parser.add_argument('-s', "--subject", type=str, default='', help='provide the subject name')
    parser.add_argument('-v', "--verbose", action="store_true")
    args = parser.parse_args()

    if (args.session1!='') and (args.session2!=''):
        update_data(args)
        # else:
        #     print('"%s" not a valid datafolder' % args.datafolder)




