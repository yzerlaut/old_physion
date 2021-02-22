import subprocess

# command = '...'

# if platform.system() == 'Windows':
#     proc = psutil.Process(os.getpid())
#     # proc.set_nice(psutil.HIGH_PRIORITY_CLASS)
#     print(dir(proc))
#     #p.set_nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
#     # proc.set_nice(psutil.IDLE_PRIORITY_CLASS)           # Sets low priority
# print(platform)

subprocess.run('python pupil\process.py -d 2021_02_16 -t 15-41-13 -s 100',
               creationflags=subprocess.HIGH_PRIORITY_CLASS)
