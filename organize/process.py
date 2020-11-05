import os

def list_TSeries_folder(folder):
    folders = [os.path.join(folder, d) for d in sorted(os.listdir(folder)) if ((d[:7]=='TSeries') and os.path.isdir(os.path.join(folder, d)))]
    return folders
