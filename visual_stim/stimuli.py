from psychopy import visual, core, event, clock, monitors # some libraries from PsychoPy
import numpy as np
import itertools, os, sys, pathlib, subprocess, time, datetime
import pynwb
from hdmf.data_utils import DataChunkIterator

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
from noise import build_sparse_noise, build_dense_noise
from natural_images import load, img_after_hist_normalization

def stop_signal(parent):
    if (len(event.getKeys())>0) or (parent.stop_flag):
        parent.stop_flag = True
        parent.statusBar.showMessage('stimulation stopped !')
        return True
    else:
        return False


class visual_stim:

    def __init__(self,
                 protocol,
                 screen = {'name':'Lilliput',
                           'screen_id':1,
                           'resolution':[1280, 768],
                           'width':16, # in cm
                           'distance_from_eye':15, # in cm
                           'monitoring_square':{'size':8,
                                                'location':'bottom-left',
                                                'color-on':1,
                                                'color-off':0,
                                                'time-on':0.2,
                                                'time-off':0.8},
                           'gamma_correction':{'k':1.03,
                                               'gamma':1.77}},
                 stimuli_folder=os.path.join(os.path.expanduser('~'), 'DATA', 'STIMULI'),
                 demo=True):
        """
        """
        
        self.protocol = protocol
        if 'filename' not in protocol:
            protocol['filename'] = protocol['Stimulus']
        self.screen = screen
        self.demo = demo
        self.stimuli_folder = stimuli_folder
        self.io, self.nwbfile = None, None

        # self.monitoring_square = monitoring_square
        # # gamma_correction
        # self.k, self.gamma = self.screen['gamma_correction']['k'], self.screen['gamma_correction']['gamma']

        
    def init_presentation(self):
        if self.demo or (self.protocol['Setup']=='demo-mode'):
            self.monitor = monitors.Monitor('testMonitor')
            self.win = visual.Window((int(screen['resolution'][0]/2),
                                      int(screen['resolution'][1]/2)),
                                     monitor=self.monitor, screen=0,
                                     units='pix', color=-1) #create a window
        else:
            self.monitor = monitors.Monitor(screen['name'])
            self.win = visual.Window((screen['resolution'][0],
                                      screen['resolution'][1]),
                                     monitor=self.monitor,
                                     screen=screen['screen_id'], fullscr=True, units='deg', color=-1)
            
    # Gamma correction 
    def gamma_corrected_lum(self, level):
        return 2*np.power(((level+1.)/2./self.screen['gamma_correction']['k']), 1./self.screen['gamma_correction']['gamma'])-1.
    
    def gamma_corrected_contrast(self, contrast):
        return np.power(contrast/self.screen['gamma_correction']['k'], 1./self.screen['gamma_correction']['gamma'])
    
    # initialize all quantities
    def init_experiment(self, protocol, keys):

        self.experiment, self.PATTERNS = {}, []

        if protocol['Presentation']=='Single-Stimulus':
            for key in protocol:
                if key.split(' (')[0] in keys:
                    self.experiment[key.split(' (')[0]] = [protocol[key]]
                    self.experiment['index'] = [0]
                    self.experiment['repeat'] = [0]
                    self.experiment['time_start'] = [protocol['presentation-prestim-period']]
                    self.experiment['time_stop'] = [protocol['presentation-duration']+protocol['presentation-prestim-period']]
        else: # MULTIPLE STIMS
            VECS, FULL_VECS = [], {}
            for key in keys:
                FULL_VECS[key], self.experiment[key] = [], []
                if protocol['N-'+key]>1:
                    VECS.append(np.linspace(protocol[key+'-1'], protocol[key+'-2'],protocol['N-'+key]))
                else:
                    VECS.append(np.array([protocol[key+'-2']]))
            for vec in itertools.product(*VECS):
                for i, key in enumerate(keys):
                    FULL_VECS[key].append(vec[i])
                    
            self.experiment['index'], self.experiment['repeat'] = [], []
            self.experiment['time_start'], self.experiment['time_stop'] = [], []

            index_no_repeat = np.arange(len(FULL_VECS[key]))

            # SHUFFLING IF NECESSARY
            if (protocol['Presentation']=='Randomized-Sequence'):
                np.random.seed(protocol['shuffling-seed'])
                np.random.shuffle(index_no_repeat)
                
            Nrepeats = max([1,protocol['N-repeat']])
            index = np.concatenate([index_no_repeat for r in range(Nrepeats)])
            repeat = np.concatenate([r+0*index_no_repeat for r in range(Nrepeats)])

            for n, i in enumerate(index[protocol['starting-index']:]):
                for key in keys:
                    self.experiment[key].append(FULL_VECS[key][i])
                self.experiment['index'].append(i)
                self.experiment['repeat'].append(repeat[n+protocol['starting-index']])
                self.experiment['time_start'].append(protocol['presentation-prestim-period']+\
                                                     n*protocol['presentation-duration']+n*protocol['presentation-interstim-period'])
                self.experiment['time_stop'].append(protocol['presentation-prestim-period']+\
                                                     (n+1)*protocol['presentation-duration']+n*protocol['presentation-interstim-period'])

        
    # the close function
    def close(self):
        if self.io is not None:
            self.io.close()
        self.win.close()

    def quit(self):
        if self.io is not None:
            self.io.close()
        core.quit()

    def check_movie(self):
        if ('movie_filename' in self.protocol) and os.path.isfile(self.protocol['movie_filename']):
            return True
        else:
            return False

    def preload_movie(self):

        if check_movie:
            self.io = pynwb.NWBHDF5IO(self.protocol['movie_filename'], 'r')
            self.nwbfile = self.io.read()

            # blinking in bottom-left corner
    def add_monitoring_signal(self, img, tnow, tstart):
        
        if (int(1e3*new_t-1e3*start)<self.Tfull) and\
           (int(1e3*new_t-1e3*start)%self.Tfull_first<self.Ton):
            self.on.draw()
        elif int(1e3*new_t-1e3*start)%self.Tfull<self.Ton:
            self.on.draw()
        else:
            self.off.draw()

    def run(self, parent):
        pass
        # self.win.show()
        if self.nwbfile is not None:
            start = clock.getTime()
            while ((clock.getTime()-start)<self.protocol['presentation-duration']) and not parent.stop_flag:
                index = int((clock.getTime()-start)/self.dt)
        else:
            print(' Need to generate and preload the nwbfile movie')
            print(' /!\ running not possible ! /!\ ')
                
            
    def generate_movie(self,
                       refresh_freq=60.):
        # has to be implemented in the child class
        print('Not implemented in for this stimulus...')
        # for i in range(1000):
        filename = os.path.join(self.stimuli_folder,
                                self.protocol['filename'].split(os.path.sep)[-1].replace('.json',
                                    '_'+datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')+'.nwb'))
        self.nwbfile = pynwb.NWBFile(identifier=filename,
                                     session_description='Movie file for stimulus presentation',
                                     session_start_time=datetime.datetime.now(),
                                     experiment_description=str(self.protocol),
                                     experimenter='Yann Zerlaut',
                                     lab='Bacci and Rebola labs',
                                     institution='Paris Brain Institute',
                                     source_script=str(pathlib.Path(__file__).resolve()),
                                     source_script_file_name=str(pathlib.Path(__file__).resolve()),
                                     file_create_date=datetime.datetime.today())

        data = DataChunkIterator(data=self.frame_generator())
        # frame_stimuli = pynwb.TimeSeries(name='visual-stimuli',
        frame_stimuli = pynwb.image.ImageSeries(name='visual-stimuli',
                                                data=data,
                                                unit='NA',
                                                starting_time=0.,
                                                rate=refresh_freq)
        self.nwbfile.add_stimulus(frame_stimuli)

        io = pynwb.NWBHDF5IO(filename, 'w')

        io.write(self.nwbfile)
        io.close()

        io = pynwb.NWBHDF5IO(filename, 'r')
        self.nwbfile = io.read()
        print(self.nwbfile.stimulus['visual-stimuli'].data.shape)
        io.close()
        
        

    def frame_generator(self):
        """
        Generator creating a random number of chunks (but at most max_chunks) of length chunk_length containing
        random samples of sin([0, 2pi]).
        """
        x = 0
        num_chunks = 0
        # tstart = time.time()
        # while (time.time()-tstart)<10:
        #     val = np.asarray([np.sin(np.random.randn() * 2 * np.pi) for i in range(chunk_length)])
        #     x = np.random.randn()
        #     num_chunks += 1
        #     yield np.ones(self.screen['resolution'])
        for i in range(1000):
            yield np.ones(self.screen['resolution'])
        return        
        
        
#####################################################
##  ----   PRESENTING VARIOUS LIGHT LEVELS  --- #####           
#####################################################

class light_level_single_stim(visual_stim):

    def __init__(self, protocol):
        
        super().__init__(protocol)
        super().init_experiment(protocol, ['light-level'])
        
        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
            visual.GratingStim(win=self.win,
                               size=1000, pos=[0,0], sf=0,
                               color=self.gamma_corrected_lum(self.experiment['light-level'][i]))])
            
#####################################################
##  ----   PRESENTING FULL FIELD GRATINGS   --- #####           
#####################################################

class full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['spatial-freq', 'angle', 'contrast'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     size=1000, pos=[0,0],
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i]))])

            
class drifting_full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['spatial-freq', 'angle', 'contrast', 'speed'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     size=1000, pos=[0,0],
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i]))])

        
#####################################################
##  ----    PRESENTING CENTERED GRATINGS    --- #####           
#####################################################

class center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius'][i], mask='circle',
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i]))])

class drifting_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'speed', 'bg-color'])

        print(self.experiment['bg-color'])
        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius'][i], mask='circle',
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i]))])


#####################################################
##  ----    PRESENTING OFF-CENTERED GRATINGS    --- #####           
#####################################################

class off_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     size=1000, pos=[0,0],
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i])),
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius'][i],
                                                     mask='circle', sf=0,
                                                     color=self.gamma_corrected_lum(self.experiment['bg-color'][i]))])

            
class drifting_off_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color', 'speed'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  # Surround grating
                                  visual.GratingStim(win=self.win,
                                                     size=1000, pos=[0,0],
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i])),
                                  # + center Mask
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius'][i],
                                                     mask='circle', sf=0, contrast=0,
                                                     color=self.gamma_corrected_lum(self.experiment['bg-color'][i]))])


#####################################################
##  ----    PRESENTING SURROUND GRATINGS    --- #####           
#####################################################
        
class surround_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius-start', 'radius-end','spatial-freq', 'angle', 'contrast', 'bg-color'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     size=1000, pos=[0,0], sf=0,
                                                     color=self.gamma_corrected_lum(self.experiment['bg-color'][i])),
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius-end'][i],
                                                     mask='circle', 
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i])),
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius-start'][i],
                                                     mask='circle', sf=0,
                                                     color=self.gamma_corrected_lum(self.experiment['bg-color'][i]))])

class drifting_surround_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius-start', 'radius-end','spatial-freq', 'angle', 'contrast', 'bg-color', 'speed'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     size=1000, pos=[0,0], sf=0,contrast=0,
                                                     color=self.gamma_corrected_lum(self.experiment['bg-color'][i])),
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius-end'][i],
                                                     mask='circle', 
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i])),
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius-start'][i],
                                                     mask='circle', sf=0,contrast=0,
                                                     color=self.gamma_corrected_lum(self.experiment['bg-color'][i]))])
        

#####################################################
##  ----    PRESENTING NATURAL IMAGES       --- #####
#####################################################

NI_directory = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'NI_bank')
        
class natural_image(visual_stim):

    def __init__(self, protocol):

        # from visual_stim.psychopy_code.preprocess_NI import load, img_after_hist_normalization
        # from .preprocess_NI import load, img_after_hist_normalization
        
        super().__init__(protocol)
        super().init_experiment(protocol, ['Image-ID'])

        for i in range(len(self.experiment['index'])):
            filename = os.listdir(NI_directory)[int(self.experiment['Image-ID'][i])]
            img = load(os.path.join(NI_directory, filename))
            img = 2*self.gamma_corrected_contrast(img_after_hist_normalization(img))-1 # normalization + gamma_correction
            # rescaled_img = adapt_to_screen_resolution(img, (SCREEN[0], SCREEN[1]))

            self.PATTERNS.append([visual.ImageStim(self.win, image=img.T,
                                                   units='pix', size=self.win.size)])


#####################################################
##  --    WITH VIRTUAL SCENE EXPLORATION    --- #####
#####################################################


def generate_VSE(duration=5,
                 mean_saccade_duration=2.,# in s
                 std_saccade_duration=1.,# in s
                 # saccade_amplitude=50.,
                 saccade_amplitude=100, # in pixels, TO BE PUT IN DEGREES
                 seed=0):
    """
    to do: clean up the VSE generator
    """
    print('generating Virtual-Scene-Exploration [...]')
    
    np.random.seed(seed)
    
    tsaccades = np.cumsum(np.clip(mean_saccade_duration+np.abs(np.random.randn(int(1.5*duration/mean_saccade_duration))*std_saccade_duration),
                                  mean_saccade_duration/2., 1.5*mean_saccade_duration))

    x = np.array(np.clip(np.random.randn(len(tsaccades))*saccade_amplitude, 0, saccade_amplitude), dtype=int)
    y = np.array(np.clip(np.random.randn(len(tsaccades))*saccade_amplitude, 0, saccade_amplitude), dtype=int)
    
    return {'t':np.array([0]+list(tsaccades)),
            'x':np.array([0]+list(x)),
            'y':np.array([0]+list(y)),
            'max_amplitude':saccade_amplitude}

            

class natural_image_vse(visual_stim):

    def __init__(self, protocol):

        super().__init__(protocol)
        super().init_experiment(protocol, ['Image-ID', 'VSE-seed', 'mean-saccade-duration', 'std-saccade-duration'])

        print(self.experiment)
        self.VSEs = [] # array of Virtual-Scene-Exploration
        for i in range(len(self.experiment['index'])):

            vse = generate_VSE(duration=protocol['presentation-duration'],
                               mean_saccade_duration=self.experiment['mean-saccade-duration'][i],
                               std_saccade_duration=self.experiment['std-saccade-duration'][i],
                               saccade_amplitude=100, # in pixels, TO BE PUT IN DEGREES
                               seed=int(self.experiment['VSE-seed'][i]+self.experiment['Image-ID'][i]))

            self.VSEs.append(vse)
            
            filename = os.listdir(NI_directory)[int(self.experiment['Image-ID'][i])]
            img = load(os.path.join(NI_directory, filename))
            img = 2*self.gamma_corrected_contrast(img_after_hist_normalization(img))-1 # normalization + gamma_correction
            # rescaled_img = adapt_to_screen_resolution(img, (SCREEN[0], SCREEN[1]))
            sx, sy = img.T.shape

            self.PATTERNS.append([])
            
            IMAGES = []
            for i in range(len(vse['t'])):
                ix, iy = vse['x'][i], vse['y'][i]
                new_im = img.T[ix:sx-vse['max_amplitude']+ix,\
                               iy:sy-vse['max_amplitude']+iy]
                self.PATTERNS[-1].append(visual.ImageStim(self.win,
                                                          image=new_im,
                                                          units='pix', size=self.win.size))
            
    

#####################################################
##  ----    PRESENTING BINARY NOISE         --- #####
#####################################################

class sparse_noise(visual_stim):
    
    def __init__(self, protocol):

        super().__init__(protocol)
        super().init_experiment(protocol,
            ['square-size', 'sparseness', 'mean-refresh-time', 'jitter-refresh-time'])
        
        self.STIM = build_sparse_noise(protocol['presentation-duration'],
                                       self.monitor,
                                       square_size=protocol['square-size (deg)'],
                                       noise_mean_refresh_time=protocol['mean-refresh-time (s)'],
                                       noise_rdm_jitter_refresh_time=protocol['jitter-refresh-time (s)'],
                                       seed=protocol['noise-seed (#)'])
        
        self.experiment = {'refresh-times':self.STIM['t']}
            

class dense_noise(visual_stim):

    def __init__(self, protocol):

        super().__init__(protocol)
        super().init_experiment(protocol,
                ['square-size', 'sparseness', 'mean-refresh-time', 'jitter-refresh-time'])

        self.STIM = build_dense_noise(protocol['presentation-duration'],
                                      self.monitor,
                                      square_size=protocol['square-size (deg)'],
                                      noise_mean_refresh_time=protocol['mean-refresh-time (s)'],
                                      noise_rdm_jitter_refresh_time=protocol['jitter-refresh-time (s)'],
                                      seed=protocol['noise-seed (#)'])

        self.experiment = {'refresh-times':self.STIM['t']}
            
def build_stim(protocol):
    """
    """
    if (protocol['Stimulus']=='light-level'):
        return light_level_single_stim(protocol)
    elif (protocol['Stimulus']=='full-field-grating'):
        return full_field_grating_stim(protocol)
    elif (protocol['Stimulus']=='center-grating'):
        return center_grating_stim(protocol)
    elif (protocol['Stimulus']=='off-center-grating'):
        return off_center_grating_stim(protocol)
    elif (protocol['Stimulus']=='surround-grating'):
        return surround_grating_stim(protocol)
    elif (protocol['Stimulus']=='drifting-full-field-grating'):
        return drifting_full_field_grating_stim(protocol)
    elif (protocol['Stimulus']=='drifting-center-grating'):
        return drifting_center_grating_stim(protocol)
    elif (protocol['Stimulus']=='drifting-off-center-grating'):
        return drifting_off_center_grating_stim(protocol)
    elif (protocol['Stimulus']=='drifting-surround-grating'):
        return drifting_surround_grating_stim(protocol)
    elif (protocol['Stimulus']=='Natural-Image'):
        return natural_image(protocol)
    elif (protocol['Stimulus']=='Natural-Image+VSE'):
        return natural_image_vse(protocol)
    elif (protocol['Stimulus']=='sparse-noise'):
        if protocol['Presentation']=='Single-Stimulus':
            return sparse_noise(protocol)
        else:
            print('Noise stim have to be done as "Single-Stimulus" !')
    elif (protocol['Stimulus']=='dense-noise'):
        if protocol['Presentation']=='Single-Stimulus':
            return dense_noise(protocol)
        else:
            print('Noise stim have to be done as "Single-Stimulus" !')
    else:
        print('Protocol not recognized !')
        return None

    


if __name__=='__main__':

    import json
    fn = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]),
                      'exp', 'protocols',
                      'center-drifting-gratings-4-orientations-5x5-locations-3-repeats.json')
    with open(fn, 'r') as fp:
        protocol = json.load(fp)
    protocol['filename'] = fn
    
    # stim = light_level_single_stim(protocol)
    stim = visual_stim(protocol, demo=True)
    stim.generate_movie()
    # stim.close()
