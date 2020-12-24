import os, sys, pathlib, shutil, time, datetime
import numpy as np
from pynwb import NWBFile

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import get_files_with_extension, list_dayfolder, check_datafolder, get_TSeries_folders

start_time = datetime.datetime(2017, 4, 3, 9, 8, 7)
create_date = datetime.datetime(2017, 4, 15, 9, 8, 7)

nwbfile = NWBFile(session_description='demonstrate NWBFile basics',  # required
                                    identifier='NWB123',  # required
                                    session_start_time=start_time,  # required
                                    file_create_date=create_date)  # optional

if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="transfer interface",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rf', "--root_datafolder", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-d', "--day", type=str,
                        default='')
    parser.add_argument('-wt', "--with_transfer", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    args = parser.parse_args()

    if args.day!='':
        folder = os.path.join(args.root_datafolder, args.day)
    else:
        folder = args.root_datafolder

    print(folder)
    # if args.day!='':
    #     PROTOCOL_LIST = list_dayfolder(os.path.join(vis_folder, args.day))
    # else: # loop over days
    #     PROTOCOL_LIST = []
    #     for day in os.listdir(vis_folder):
    #         PROTOCOL_LIST += list_dayfolder(os.path.join(vis_folder, day))
    #     print(PROTOCOL_LIST)
    # CA_FILES = find_matching_data(PROTOCOL_LIST, CA_FILES,
    #                               verbose=args.verbose)

    # if args.with_transfer:
    #     transfer_analyzed_data(CA_FILES)
