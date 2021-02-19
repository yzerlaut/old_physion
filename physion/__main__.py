from . import run
if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="Main script",
                                   formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rf', "--root_datafolder", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    args = parser.parse_args()
    run(args)

if __name__=='__main__':
    run()


