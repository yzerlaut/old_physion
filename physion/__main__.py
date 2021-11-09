import argparse, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from physion import run

if __name__=='__main__':
    import argparse, os
    parser=argparse.ArgumentParser(description="Physion",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rf', "--root_datafolder", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-d', "--demo", action="store_true")
    args = parser.parse_args()
    run(args)


