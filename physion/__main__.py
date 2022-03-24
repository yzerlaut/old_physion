import argparse, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from physion import run

if __name__=='__main__':
    import argparse, os
    parser=argparse.ArgumentParser(description="Physion",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rf', "--root_datafolder", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-f', "--file")
    parser.add_argument('-sd', "--stim_demo", action="store_true")
    parser.add_argument('-d', "--demo", action="store_true")
    args = parser.parse_args()
    
    if args.stim_demo:

        if os.path.isfile(args.file):
            from physion.visual_stim.psychopy_code.stimuli import json, build_stim, dummy_parent

            # load protocol
            with open(args.file, 'r') as fp:
                protocol = json.load(fp)
            protocol['demo'] = True
            # launch protocol
            stim = build_stim(protocol)
            parent = dummy_parent()
            stim.run(parent)
            stim.close()
        else:
            print('Need to provide a valid (json) stimulus file as a "--file" argument ! ')

    else:
        run(args)


