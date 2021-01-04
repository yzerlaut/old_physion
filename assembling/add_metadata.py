"""
Script to add/modify the metadata in case of mistakes
"""
import os, tempfile, json, pathlib, shutil
import numpy as np

base_path = str(pathlib.Path(__file__).resolve().parents[1])

def add_metadata(args):

    fn = os.path.join(args.datafolder, 'metadata.npy')

    # load previous
    metadata = np.load(fn, allow_pickle=True).item()
    temp = str(tempfile.NamedTemporaryFile().name)+'.npy'
    print("""
    ---> moving the old metadata to the temporary file directory as: "%s" [...]
    """ % temp)
    shutil.move(fn, temp)

    # updates of config
    if args.config!='':
        try:
            with open(args.config) as f:
                config = json.load(f)
            metadata['config'] = args.config.split(os.path.sep)[-1].replace('.json', '')
            for key in config:
                metadata[key] = config[key]
        except BaseException as be:
            print(be)
            print(' /!\ update of "Config" metadata failed /!\ ')

    # updates of protocol
    if args.protocol!='':
        try:
            with open(args.protocol) as f:
                protocol = json.load(f)
            metadata['protocol'] = args.protocol.split(os.path.sep)[-1].replace('.json', '')
            for key in protocol:
                metadata[key] = protocol[key]
        except BaseException as be:
            print(be)
            print(' /!\ update of "Protocol" metadata failed /!\ ')
            
    # updates of subject
    if args.subject!='':
        try:
            with open(args.subject_file) as f:
                subjects = json.load(f)
            metadata['subject_ID'] = args.subject
            metadata['subject_props'] = subjects[args.subject]
        except BaseException as be:
            print(be)
            print(' /!\ update of "Subject" metadata failed /!\ ')
    
    # save new
    if 'notes' not in metadata:
        metadata['notes'] = ''
        
    np.save(fn, metadata)


if __name__=='__main__':

    import argparse
    parser=argparse.ArgumentParser(description="""
    Building NWB file from mutlimodal experimental recordings
    """,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-df', "--datafolder", type=str, default='')
    parser.add_argument('-c', "--config", type=str, default='', help='provide the full path !')
    parser.add_argument('-p', "--protocol", type=str, default='', help='provide the full path !')
    parser.add_argument('-sf', "--subject_file", type=str,
        default=os.path.join(base_path, 'exp', 'subjects.json'))
    parser.add_argument('-s', "--subject", type=str, default='', help='provide the subject name')
    parser.add_argument('-v', "--verbose", action="store_true")
    args = parser.parse_args()

    if args.datafolder!='':
        if os.path.isdir(args.datafolder):
            add_metadata(args)
        else:
            print('"%s" not a valid datafolder' % args.datafolder)
