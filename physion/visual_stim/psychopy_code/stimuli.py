"""
N.B. there is a mix of "grey screen" corresponding to bg-color=0 and bg-color=0.5 (see in individual protocols) TO BE FIXED
"""
import numpy as np
import itertools, os, sys, pathlib, time, json
try:
    from psychopy import visual, core, event, clock, monitors # We actually do it below so that we can use the code without psychopy
except (ModuleNotFoundError, ImportError):
    pass
 
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from screens import SCREENS
from psychopy_code.noise import sparse_noise_generator, dense_noise_generator
from psychopy_code.preprocess_NI import load, img_after_hist_normalization, adapt_to_screen_resolution

def build_stim(protocol, no_psychopy=False):
    """
    """
    if not no_psychopy:
        from psychopy import visual, core, event, clock, monitors # some libraries from PsychoPy
    
    if (protocol['Presentation']=='multiprotocol'):
        return multiprotocol(protocol, no_psychopy=no_psychopy)
    elif (protocol['Stimulus']=='light-level'):
        return light_level_single_stim(protocol)
    elif (protocol['Stimulus']=='bar'):
        return bar_stim(protocol)
    elif (protocol['Stimulus']=='full-field-grating'):
        return full_field_grating_stim(protocol)
    elif (protocol['Stimulus']=='oddball-full-field-grating'):
        return oddball_full_field_grating_stim(protocol)
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
    elif (protocol['Stimulus']=='line-moving-dots'):
        return line_moving_dots(protocol)
    elif (protocol['Stimulus']=='looming-stim'):
        return looming_stim(protocol)
    elif (protocol['Stimulus']=='mixed-moving-dots-static-patch'):
        return moving_dots_static_patch(protocol)
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
    elif (protocol['Stimulus']=='gaussian-blobs'):
        return gaussian_blobs(protocol)
    else:
        print('Protocol not recognized !')
        return None

    
def stop_signal(parent):
    if (len(event.getKeys())>0) or (parent.stop_flag):
        parent.stop_flag = True
        if hasattr(parent, 'statusBar'):
            parent.statusBar.showMessage('stimulation stopped !')
        return True
    else:
        return False


class visual_stim:

    def __init__(self, protocol, demo=False, store_frame=False):
        """
        """
        self.protocol = protocol
        self.store_frame = store_frame
        if ('store_frame' in protocol):
            self.store_frame = bool(protocol['store_frame'])

        self.screen = SCREENS[self.protocol['Screen']]

        # we can initialize the angle
        self.x, self.z = self.angle_meshgrid()

        if not ('no-window' in self.protocol):

            self.monitor = monitors.Monitor(self.screen['name'])
            self.monitor.setDistance(self.screen['distance_from_eye'])
            
            if demo or (('demo' in self.protocol) and self.protocol['demo']):
                # we override the parameters
                self.screen['monitoring_square']['size'] = int(600*self.screen['monitoring_square']['size']/self.screen['resolution'][0])
                self.screen['resolution'] = (600,int(600*self.screen['resolution'][1]/self.screen['resolution'][0]))
                self.screen['screen_id'] = 0
                self.screen['fullscreen'] = False
                self.protocol['movie_refresh_freq'] = 5.

            self.k, self.gamma = self.screen['gamma_correction']['k'], self.screen['gamma_correction']['gamma']

            self.win = visual.Window(self.screen['resolution'], monitor=self.monitor,
                                     screen=self.screen['screen_id'], fullscr=self.screen['fullscreen'],
                                     units='pix',
                                     color=self.gamma_corrected_lum(self.protocol['presentation-prestim-screen']))

            # blank screens
            self.blank_start = visual.GratingStim(win=self.win, size=10000, pos=[0,0], sf=0,
                                                  color=self.gamma_corrected_lum(self.protocol['presentation-prestim-screen']),
                                                  units='pix')
            if 'presentation-interstim-screen' in self.protocol:
                self.blank_inter = visual.GratingStim(win=self.win, size=10000, pos=[0,0], sf=0,
                                                      color=self.gamma_corrected_lum(self.protocol['presentation-interstim-screen']),
                                                      units='pix')
            self.blank_end = visual.GratingStim(win=self.win, size=10000, pos=[0,0], sf=0,
                                                color=self.gamma_corrected_lum(self.protocol['presentation-poststim-screen']),
                                                units='pix')

            if self.screen['monitoring_square']['location']=='top-right':
                pos = [int(x/2.-self.screen['monitoring_square']['size']/2.) for x in self.screen['resolution']]
            elif self.screen['monitoring_square']['location']=='bottom-left':
                pos = [int(-x/2.+self.screen['monitoring_square']['size']/2.) for x in self.screen['resolution']]
            elif self.screen['monitoring_square']['location']=='top-left':
                pos = [int(-self.screen['resolution'][0]/2.+self.screen['monitoring_square']['size']/2.),
                       int(self.screen['resolution'][1]/2.-self.screen['monitoring_square']['size']/2.)]
            elif self.screen['monitoring_square']['location']=='bottom-right':
                pos = [int(self.screen['resolution'][0]/2.-self.screen['monitoring_square']['size']/2.),
                       int(-self.screen['resolution'][1]/2.+self.screen['monitoring_square']['size']/2.)]
            else:
                print(30*'-'+'\n /!\ monitoring square location not recognized !!')

            self.on = visual.GratingStim(win=self.win, size=self.screen['monitoring_square']['size'], pos=pos, sf=0,
                                         color=self.screen['monitoring_square']['color-on'], units='pix')
            self.off = visual.GratingStim(win=self.win, size=self.screen['monitoring_square']['size'],  pos=pos, sf=0,
                                          color=self.screen['monitoring_square']['color-off'], units='pix')

            # initialize the times for the monitoring signals
            self.Ton = int(1e3*self.screen['monitoring_square']['time-on'])
            self.Toff = int(1e3*self.screen['monitoring_square']['time-off'])
            self.Tfull, self.Tfull_first = int(self.Ton+self.Toff), int((self.Ton+self.Toff)/2.)

        
    ################################
    #  ---   Gamma correction  --- #
    ################################
    def gamma_corrected_lum(self, level):
        return 2*np.power(((level+1.)/2./self.k), 1./self.gamma)-1.
    
    def gamma_corrected_contrast(self, contrast):
        return np.power(contrast/self.k, 1./self.gamma)
    
    
    ################################
    #  ---       Geometry      --- #
    ################################
    def pixel_meshgrid(self):
        return np.meshgrid(np.arange(self.screen['resolution'][0]),
                           np.arange(self.screen['resolution'][1]))
    
    def cm_to_angle(self, value):
        return 180./np.pi*np.arctan(value/self.screen['distance_from_eye'])
    
    def pix_to_angle(self, value):
        return self.cm_to_angle(value/self.screen['resolution'][0]*self.screen['width'])
    
    def angle_meshgrid(self):
        x = np.linspace(self.cm_to_angle(-self.screen['width']/2.),
                        self.cm_to_angle(self.screen['width']/2.),
                        self.screen['resolution'][0])
        z = np.linspace(self.cm_to_angle(-self.screen['width']/2.)*self.screen['resolution'][1]/self.screen['resolution'][0],
                        self.cm_to_angle(self.screen['width']/2.)*self.screen['resolution'][1]/self.screen['resolution'][0],
                        self.screen['resolution'][1])
        # X, Z = np.meshgrid(x, z)
        # return np.rot90(X, k=3).T, np.rot90(Z, k=3).T
        return np.meshgrid(x, z)

    def angle_to_cm(self, value):
        return self.screen['distance_from_eye']*np.tan(np.pi/180.*value)
    
    def angle_to_pix(self, value, from_x_center=False, from_z_center=False, starting_angle=0):
        """
        We deal here with the non-linear transformation of angle to distance on the screen (see "tan" function toward 90deg)
        we introduce a "starting_angle" so that a size-on-screen can be taken 
        """
        # if from_x_center:
        #     return self.screen['resolution'][0]/self.screen['width']*self.angle_to_cm(value)+self.screen['resolution'][0]/2.
        # elif from_z_center:
        #     return self.screen['resolution'][0]/self.screen['width']*self.angle_to_cm(value)+\
        #         self.screen['resolution'][0]*self.screen['height']/self.screen['width']/2.
        if starting_angle==0:
            return self.screen['resolution'][0]/self.screen['width']*self.angle_to_cm(value)
        else:
            pix1 = self.screen['resolution'][0]/self.screen['width']*self.angle_to_cm(np.abs(starting_angle)+value)
            pix2 = self.screen['resolution'][0]/self.screen['width']*self.angle_to_cm(np.abs(starting_angle))
            return np.abs(pix2-pix1)

                           
    ################################
    #  ---     Experiment      --- #
    ################################

    def init_experiment(self, protocol, keys, run_type='static'):

        self.experiment, self.PATTERNS = {}, []

        if protocol['Presentation']=='Single-Stimulus':
            for key in protocol:
                if key.split(' (')[0] in keys:
                    self.experiment[key.split(' (')[0]] = [protocol[key]]
                    self.experiment['index'] = [0]
                    self.experiment['frame_run_type'] = [run_type]
                    self.experiment['index'] = [0]
                    self.experiment['time_start'] = [protocol['presentation-prestim-period']]
                    self.experiment['time_stop'] = [protocol['presentation-duration']+protocol['presentation-prestim-period']]
                    self.experiment['time_duration'] = [protocol['presentation-duration']]
                    self.experiment['interstim'] = [protocol['presentation-interstim-period'] if 'presentation-interstim-period' in protocol else 0]
                    self.experiment['interstim-screen'] = [protocol['presentation-interstim-screen'] if 'presentation-interstim-screen' in protocol else 0]
        else: # MULTIPLE STIMS
            VECS, FULL_VECS = [], {}
            for key in keys:
                FULL_VECS[key], self.experiment[key] = [], []
                if ('N-log-'+key in protocol) and (protocol['N-log-'+key]>1):
                    VECS.append(np.logspace(np.log10(protocol[key+'-1']), np.log10(protocol[key+'-2']),protocol['N-log-'+key]))
                elif protocol['N-'+key]>1:
                    VECS.append(np.linspace(protocol[key+'-1'], protocol[key+'-2'],protocol['N-'+key]))
                else:
                    VECS.append(np.array([protocol[key+'-2']])) # we pick the SECOND VALUE as the constant one (so remember to fill this right in GUI)
            for vec in itertools.product(*VECS):
                for i, key in enumerate(keys):
                    FULL_VECS[key].append(vec[i])
                    
            self.experiment['index'], self.experiment['repeat'] = [], []
            self.experiment['time_start'], self.experiment['time_stop'] = [], []
            self.experiment['interstim'], self.experiment['time_duration'] = [], [] # relevant for multi-protocols
            self.experiment['interstim-screen'], self.experiment['frame_run_type'] = [], []

            index_no_repeat = np.arange(len(FULL_VECS[key]))

            # then dealing with repetitions
            Nrepeats = max([1,protocol['N-repeat']])

            if 'shuffling-seed' in protocol:
                np.random.seed(protocol['shuffling-seed']) # initialize random seed

            for r in range(Nrepeats):
                
                # shuffling if necessary !
                if (protocol['Presentation']=='Randomized-Sequence'):
                    np.random.shuffle(index_no_repeat)
                
                for n, i in enumerate(index_no_repeat):
                    for key in keys:
                        self.experiment[key].append(FULL_VECS[key][i])
                    self.experiment['index'].append(i) # shuffled
                    self.experiment['repeat'].append(r)
                    self.experiment['time_start'].append(protocol['presentation-prestim-period']+\
                                                         (r*len(index_no_repeat)+n)*\
                                                         (protocol['presentation-duration']+protocol['presentation-interstim-period']))
                    self.experiment['time_stop'].append(self.experiment['time_start'][-1]+protocol['presentation-duration'])
                    self.experiment['interstim'].append(protocol['presentation-interstim-period'])
                    self.experiment['interstim-screen'].append(protocol['presentation-interstim-screen'])
                    self.experiment['time_duration'].append(protocol['presentation-duration'])
                    self.experiment['frame_run_type'].append(run_type)
                    
                    
    # the close function
    def close(self):
        self.win.close()

    def quit(self):
        core.quit()

    # screen at start
    def start_screen(self, parent):
        if not parent.stop_flag:
            self.blank_start.draw()
            self.off.draw()
            try:
                self.win.flip()
                if self.store_frame:
                    self.win.getMovieFrame() # we store the last frame
            except AttributeError:
                pass
            clock.wait(self.protocol['presentation-prestim-period'])

    # screen at end
    def end_screen(self, parent):
        if not parent.stop_flag:
            self.blank_end.draw()
            self.off.draw()
            try:
                self.win.flip()
                if self.store_frame:
                    self.win.getMovieFrame() # we store the last frame
            except AttributeError:
                pass
            clock.wait(self.protocol['presentation-poststim-period'])

    # screen for interstim
    def inter_screen(self, parent, duration=1., color=0):
        if not parent.stop_flag and hasattr(self, 'blank_inter') and duration>0:
            visual.GratingStim(win=self.win, size=10000, pos=[0,0], sf=0,
                               color=self.gamma_corrected_lum(color), units='pix').draw()
            self.off.draw()
            try:
                self.win.flip()
                if self.store_frame:
                    self.win.getMovieFrame() # we store the last frame
            except AttributeError:
                pass
            clock.wait(duration)

    # blinking in one corner
    def add_monitoring_signal(self, new_t, start):
        """ Pulses of length Ton at the times : [0, 0.5, 1, 2, 3, 4, ...] """
        if (int(1e3*new_t-1e3*start)<self.Tfull) and (int(1e3*new_t-1e3*start)%self.Tfull_first<self.Ton):
            self.on.draw()
        elif int(1e3*new_t-1e3*start)%self.Tfull<self.Ton:
            self.on.draw()
        else:
            self.off.draw()

    def add_monitoring_signal_sp(self, new_t, start):
        """ Single pulse monitoring signal (see array_run) """
        if (int(1e3*new_t-1e3*start)<self.Ton):
            self.on.draw()
        else:
            self.off.draw()

    ##########################################################
    #############      PRESENTING STIMULI    #################
    ##########################################################
    
    #####################################################
    # showing a single static pattern
    def single_static_pattern_presentation(self, parent, index):
        start = clock.getTime()
        patterns = self.get_patterns(index)
        while ((clock.getTime()-start)<self.experiment['time_duration'][index]) and not parent.stop_flag:
            for pattern in patterns:
                pattern.draw()
            if (int(1e3*clock.getTime()-1e3*start)<self.Tfull) and\
               (int(1e3*clock.getTime()-1e3*start)%self.Tfull_first<self.Ton):
                self.on.draw()
            elif int(1e3*clock.getTime()-1e3*start)%self.Tfull<self.Ton:
                self.on.draw()
            else:
                self.off.draw()
            try:
                self.win.flip()
            except AttributeError:
                pass

    #######################################################
    # showing a single dynamic pattern with a phase advance
    def single_dynamic_grating_presentation(self, parent, index):
        start, prev_t = clock.getTime(), clock.getTime()
        patterns = self.get_patterns(index)
        self.speed = self.experiment['speed'][index]
        while ((clock.getTime()-start)<self.experiment['time_duration'][index]) and not parent.stop_flag:
            new_t = clock.getTime()
            for pattern in patterns:
                pattern.setPhase(self.speed*(new_t-prev_t), '+') # advance phase
                pattern.draw()
            self.add_monitoring_signal(new_t, start)
            prev_t = new_t
            try:
                self.win.flip()
            except AttributeError:
                pass
        self.win.getMovieFrame() # we store the last frame

        
    #####################################################
    # adding a run purely define by an array (time, x, y), see e.g. sparse_noise initialization
    def single_array_presentation(self, parent, index):
        pattern = visual.ImageStim(self.win,
                                   image=self.gamma_corrected_lum(self.get_frame(index)),
                                   units='pix', size=self.win.size)
        start = clock.getTime()
        while ((clock.getTime()-start)<(self.experiment['time_duration'][index])) and not parent.stop_flag:
            pattern.draw()
            self.add_monitoring_signal_sp(clock.getTime(), start)
            try:
                self.win.flip()
            except AttributeError:
                pass


    #####################################################
    # adding a run purely define by an array (time, x, y), see e.g. sparse_noise initialization
    def single_array_sequence_presentation(self, parent, index):
        time_indices, frames = self.get_frames_sequence(index)
        FRAMES = []
        for frame in frames:
            FRAMES.append(visual.ImageStim(self.win,
                                           image=self.gamma_corrected_lum(frame),
                                           units='pix', size=self.win.size))
        start = clock.getTime()
        while ((clock.getTime()-start)<(self.experiment['time_duration'][index])) and not parent.stop_flag:
            iframe = int((clock.getTime()-start)*self.frame_refresh)
            FRAMES[time_indices[iframe]].draw()
            self.add_monitoring_signal(clock.getTime(), start)
            try:
                self.win.flip()
            except AttributeError:
                pass


    def single_episode_run(self, parent, index):
        
        if self.experiment['frame_run_type'][index]=='drifting':
            self.single_dynamic_grating_presentation(parent, index)
        elif self.experiment['frame_run_type'][index]=='image':
            self.single_array_presentation(parent, index)
        elif self.experiment['frame_run_type'][index]=='images_sequence':
            self.single_array_sequence_presentation(parent, index)
        else: # static by defaults
            self.single_static_pattern_presentation(parent, index)
            
        # we store the last frame if needed
        if self.store_frame:
            self.win.getMovieFrame() 

        
    ## FINAL RUN FUNCTION
    def run(self, parent):
        try:
            t0 = np.load(os.path.join(str(parent.datafolder.get()), 'NIdaq.start.npy'))[0]
        except FileNotFoundError:
            print(str(parent.datafolder.get()), 'NIdaq.start.npy', 'not found !')
            t0 = time.time()
        self.start_screen(parent)
        for i in range(len(self.experiment['index'])):
            if stop_signal(parent):
                break
            t = time.time()-t0
            print('t=%.2dh:%.2dm:%.2ds - Running protocol of index %i/%i' % (t/3600, (t%3600)/60, (t%60),
                                                                       i+1, len(self.experiment['index'])))
            self.single_episode_run(parent, i)
            if i<(len(self.experiment['index'])-1):
                self.inter_screen(parent,
                                  duration=self.experiment['interstim'][i],
                                  color=self.experiment['interstim-screen'][i])
        self.end_screen(parent)
        if not parent.stop_flag and hasattr(parent, 'statusBar'):
            parent.statusBar.showMessage('stimulation over !')
        
        if self.store_frame:
            self.win.saveMovieFrames(os.path.join(str(parent.datafolder.get()),
                                                  'screen-frames', 'frame.tiff'))
    
    ##########################################################
    #############    DRAWING STIMULI (offline)  ##############
    ##########################################################
    
    def get_image(self, episode, time_from_episode_start=0, parent=None):
        print('to be implemented in child class')
        return 0*self.x
    
    def plot_stim_picture(self, episode,
                          ax=None, parent=None,
                          label=None, enhance=False):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode,
                             ax=ax,
                             label=label,
                             enhance=enhance,
                             parent=parent)

        return ax

        

    def get_prestim_image(self):
        return (1+self.protocol['presentation-prestim-screen'])/2.+0*self.x
    def get_interstim_image(self):
        return (1+self.protocol['presentation-interstim-screen'])/2.+0*self.x
    def get_poststim_image(self):
        return (1+self.protocol['presentation-poststim-screen'])/2.+0*self.x

    def show_frame(self, episode, 
                   time_from_episode_start=0,
                   parent=None,
                   label={'degree':5,
                          'shift_factor':0.02,
                          'lw':2, 'fontsize':12},
                   arrow=None,
                   vse=None,
                   enhance=False,
                   ax=None):
        """

        display the visual stimulus at a given time in a given episode of a stimulation pattern

        --> optional with angular label (switch to None to remove)
                   label={'degree':5,
                          'shift_factor':0.02,
                          'lw':2, 'fontsize':12},
        --> optional with arrow for direction propagation (switch to None to remove)
                   arrow={'direction':90,
                          'center':(0,0),
                          'length':10,
                          'width_factor':0.05,
                          'color':'red'},
        --> optional with virtual scene exploration trajectory (switch to None to remove)
        """
        
        if ax==None:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots(1)

        if enhance:
            width=80 # degree
            self.x, self.z = np.meshgrid(np.linspace(-width, width, self.screen['resolution'][0]),
                                         np.linspace(-width*self.screen['resolution'][1]/self.screen['resolution'][0],
                                                     width*self.screen['resolution'][1]/self.screen['resolution'][0],
                                                     self.screen['resolution'][1]))

        cls = (parent if parent is not None else self)

        ax.imshow(cls.get_image(episode,
                                 time_from_episode_start=time_from_episode_start,
                                 parent=cls),
                  cmap='gray', vmin=0, vmax=1, aspect='equal', origin='lower')
        
        ax.axis('off')

        if label is not None:
            nz, nx = self.x.shape
            L, shift = nx/(self.x[0][-1]-self.x[0][0])*label['degree'], label['shift_factor']*nx
            ax.plot([-shift, -shift], [-shift,L-shift], 'k-', lw=label['lw'])
            ax.plot([-shift, L-shift], [-shift,-shift], 'k-', lw=label['lw'])
            ax.annotate('%.0f$^o$ ' % label['degree'], (-shift, -shift), fontsize=label['fontsize'], ha='right', va='bottom')

        return ax


    def add_arrow(self, arrow, ax):
        nz, nx = self.x.shape
        ax.arrow(self.angle_to_pix(arrow['center'][0])+nx/2,
                 self.angle_to_pix(arrow['center'][1])+nz/2,
                 np.cos(np.pi/180.*arrow['direction'])*self.angle_to_pix(arrow['length']),
                 -np.sin(np.pi/180.*arrow['direction'])*self.angle_to_pix(arrow['length']),
                 width=self.angle_to_pix(arrow['length'])*arrow['width_factor'],
                 color=arrow['color'])
        
    def add_vse(self, arrow, ax):
        """ to be written """
        pass

    
#####################################################
##  ----         MULTI-PROTOCOLS            --- #####           
#####################################################

class multiprotocol(visual_stim):

    def __init__(self, protocol, no_psychopy=False):
        
        super().__init__(protocol)

        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 30.
        if 'appearance_threshold' not in protocol:
            protocol['appearance_threshold'] = 2.5 # 
        self.frame_refresh = protocol['movie_refresh_freq']
        
        self.STIM, i = [], 1

        if ('load_from_protocol_data' in protocol) and protocol['load_from_protocol_data']:
            while 'Protocol-%i'%i in protocol:
                subprotocol = {'Screen':protocol['Screen'],
                               'Presentation':'',
                               'no-window':True}
                for key in protocol:
                    if ('Protocol-%i-'%i in key):
                        subprotocol[key.replace('Protocol-%i-'%i, '')] = protocol[key]
                self.STIM.append(build_stim(subprotocol, no_psychopy=no_psychopy))
                i+=1
        else:
            while 'Protocol-%i'%i in protocol:
                Ppath = os.path.join(str(pathlib.Path(__file__).resolve().parents[2]), 'exp', 'protocols', protocol['Protocol-%i'%i])
                if not os.path.isfile(Ppath):
                    print(' /!\ "%s" not found in Protocol folder /!\  ' % protocol['Protocol-%i'%i])
                with open(Ppath, 'r') as fp:
                    subprotocol = json.load(fp)
                    subprotocol['Screen'] = protocol['Screen']
                    subprotocol['no-window'] = True
                    self.STIM.append(build_stim(subprotocol, no_psychopy=no_psychopy))
                    for key, val in subprotocol.items():
                        protocol['Protocol-%i-%s'%(i,key)] = val
                i+=1

        self.experiment = {'protocol_id':[]}
        # we initialize the keys
        for stim in self.STIM:
            for key in stim.experiment:
                self.experiment[key] = []
        # then we iterate over values
        for IS, stim in enumerate(self.STIM):
            for i in range(len(stim.experiment['index'])):
                for key in self.experiment:
                    if (key in stim.experiment):
                        self.experiment[key].append(stim.experiment[key][i])
                    elif key in ['interstim-screen']:
                        self.experiment[key].append(0) # if not in keys, mean 0 interstim (e.g. sparse noise stim.)
                    elif key not in ['protocol_id', 'time_duration']:
                        self.experiment[key].append(None)
                self.experiment['protocol_id'].append(IS)
                
        # # SHUFFLING IF NECESSARY
        
        if (protocol['shuffling']=='full'):
            # print('full shuffling of multi-protocol sequence !')
            np.random.seed(protocol['shuffling-seed']) # initializing random seed
            indices = np.arange(len(self.experiment['index']))
            np.random.shuffle(indices)
            
            for key in self.experiment:
                self.experiment[key] = np.array(self.experiment[key])[indices]

        if (protocol['shuffling']=='per-repeat'):
            # TO BE TESTED
            indices = np.arange(len(self.experiment['index']))
            new_indices = []
            for r in np.unique(self.experiment['repeat']):
                repeat_cond = np.argwhere(self.experiment['repeat']==r).flatten()
                r_indices = indices[repeat_cond]
                np.random.shuffle(r_indices)
                new_indices = np.concatenate([new_indices, r_indices])
                
            for key in self.experiment:
                self.experiment[key] = np.array(self.experiment[key])[new_indices]
                
        # we rebuild time
        self.experiment['time_start'][0] = protocol['presentation-prestim-period']
        self.experiment['time_stop'][0] = protocol['presentation-prestim-period']+self.experiment['time_duration'][0]
        self.experiment['interstim'] = np.concatenate([self.experiment['interstim'][1:],[self.experiment['interstim'][0]]])
        for i in range(1, len(self.experiment['index'])):
            self.experiment['time_start'][i] = self.experiment['time_stop'][i-1]+self.experiment['interstim'][i]
            self.experiment['time_stop'][i] = self.experiment['time_start'][i]+self.experiment['time_duration'][i]

        # for key in ['protocol_id', 'index', 'repeat', 'interstim', 'time_start', 'time_stop', 'time_duration']:
        #     print(self.experiment[key], key)
            
    # functions implemented in child class
    def get_frame(self, index):
        return self.STIM[self.experiment['protocol_id'][index]].get_frame(index, parent=self)
    def get_patterns(self, index):
        return self.STIM[self.experiment['protocol_id'][index]].get_patterns(index, parent=self)
    def get_frames_sequence(self, index):
        return self.STIM[self.experiment['protocol_id'][index]].get_frames_sequence(index, parent=self)
    def get_image(self, episode, time_from_episode_start=0, parent=None):
        return self.STIM[self.experiment['protocol_id'][episode]].get_image(episode, time_from_episode_start=time_from_episode_start, parent=self)
    def plot_stim_picture(self, episode, ax=None, parent=None, enhance=False):
        return self.STIM[self.experiment['protocol_id'][episode]].plot_stim_picture(episode, ax=ax, parent=self, enhance=enhance)
    # def show_interstim(self):
        


#####################################################
##  ----   PRESENTING VARIOUS LIGHT LEVELS  --- #####           
#####################################################

class light_level_single_stim(visual_stim):

    def __init__(self, protocol):
        
        super().__init__(protocol)
        super().init_experiment(protocol, ['light-level'], run_type='static')
            
    def get_patterns(self, index, parent=None):
        cls = (parent if parent is not None else self)
        return [visual.GratingStim(win=cls.win,
                                   size=10000, pos=[0,0], sf=0, units='pix',
                                   color=cls.gamma_corrected_lum(cls.experiment['light-level'][index]))]
    
    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        return 0*self.x+(1+cls.experiment['light-level'][episode])/2.

    # def plot_stim_picture(self, episode, ax=None, parent=None):
    #     ax.imshow(self.get_image(episode, parent=parent),
    #               cmap='gray', vmin=0, vmax=1, aspect='equal', origin='lower')
    

################################################
##  ----   PRESENTING VARIOUS BARS   ---     ###
#                (for intrinsic imaging maps)  #
################################################

class bar_stim(visual_stim):

    def __init__(self, protocol):
        
        super().__init__(protocol)
        super().init_experiment(protocol, ['orientation', 'width', 'degree'],
                                run_type='static')
        
            
    def get_patterns(self, index, parent=None):
        cls = (parent if parent is not None else self)
        if cls.experiment['orientation'][index]==[90]:
            size=(cls.angle_to_pix(cls.experiment['width'][index]), 2000)
            position = (cls.angle_to_pix(cls.experiment['degree'][index]), 0)
        else:
            size=(2000, cls.angle_to_pix(cls.experiment['width'][index]))
            position = (0, cls.angle_to_pix(cls.experiment['degree'][index]))
            
        return [visual.Rect(win=cls.win,
                            size=size, pos=position, units='pix',
                            fillColor=1, color=-1)]
    
    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        return 0*self.x 

    # def plot_stim_picture(self, episode, ax=None, parent=None):
    #     ax.imshow(self.get_image(episode, parent=parent),
    #               cmap='gray', vmin=0, vmax=1, aspect='equal', origin='lower')
    


#####################################################
##  ----   PRESENTING FULL FIELD GRATINGS   --- #####           
#####################################################

# some general grating functions
def compute_xrot(x, z, angle,
                 xcenter=0, zcenter=0):
    return (x-xcenter)*np.cos(angle/180.*np.pi)+(z-zcenter)*np.sin(angle/180.*np.pi)

def compute_grating(xrot,
                    spatial_freq=0.1, contrast=1, time_phase=0.):
    return contrast*(1+np.cos(2*np.pi*(spatial_freq*xrot-time_phase)))/2.


class full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol,
                                ['spatial-freq', 'angle', 'contrast'],
                                run_type='static')

    def get_patterns(self, index, parent=None):
        cls = (parent if parent is not None else self)
        return [visual.GratingStim(win=cls.win,
                                   size=10000, pos=[0,0], units='pix',
                                   sf=1./cls.angle_to_pix(1./cls.experiment['spatial-freq'][index]),
                                   ori=cls.experiment['angle'][index],
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index]))]

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        xrot = compute_xrot(cls.x, cls.z, cls.experiment['angle'][episode])
        return compute_grating(xrot,
                               spatial_freq=cls.experiment['spatial-freq'][episode],
                               contrast=cls.experiment['contrast'][episode],
                               time_phase=cls.experiment['speed'][episode]*time_from_episode_start)


    
class oddball_full_field_grating_stim(visual_stim):

    def linear_prob(self, rdm, protocol):
        N = protocol['N_deviant (#)']+protocol['N_redundant (#)']-\
            protocol['Nmin-successive-redundant (#)']
        array = np.cumsum(np.arange(1, N+1))
        return np.argwhere(rdm<=array/array[-1])[0][0]

        
    def __init__(self, protocol):

        protocol['presentation-interstim-screen'] = protocol['presentation-interstim-screen (lum.)']
        super().__init__(protocol)

        # hard coding of the sequence
        N = int(protocol['N_deviant (#)']+protocol['N_redundant (#)'])
        Ntot = int(protocol['N_repeat (#)']*N)
        self.experiment = {'index':np.arange(Ntot)%N,
                           'repeat':np.array(np.arange(Ntot)/N, dtype=int),
                           'frame_run_type':['static' for n in range(Ntot)],
                           'angle':np.ones(Ntot)*protocol['angle-redundant (deg)'],
                           'spatial-freq':np.ones(Ntot)*protocol['spatial-freq (cycle/deg)'],
                           'contrast':np.ones(Ntot)*protocol['contrast (norm.)']}
        
        deviant_episode = []
        for i, rdm in enumerate(np.random.uniform(0,1,size=int(protocol['N_repeat (#)']))):
            inew = int(i*N+protocol['Nmin-successive-redundant (#)']+\
                       self.linear_prob(rdm, protocol))
            self.experiment['angle'][inew] = protocol['angle-deviant (deg)']
            deviant_episode.append(inew%N)

        # # TO CHECK THE LINEAR DISTRIB OF TIME
        # import matplotlib.pylab as plt
        # plt.hist(deviant_episode, bins = np.arange(N+1))
        # plt.show()

        self.experiment['interstim'] = protocol['mean-interstim (s)']+\
            np.random.randn(Ntot)*protocol['jitter-interstim (s)']
        self.experiment['time_start'] = np.cumsum(\
                np.concatenate([[protocol['presentation-prestim-period']],\
                                protocol['stim-duration (s)']+self.experiment['interstim'][:-1]]))
        self.experiment['time_stop'] = protocol['stim-duration (s)']+self.experiment['time_start']
        self.experiment['time_duration'] = protocol['stim-duration (s)']*np.ones(Ntot)



    def get_patterns(self, index, parent=None):
        cls = (parent if parent is not None else self)
        return [visual.GratingStim(win=cls.win,
                                   size=10000, pos=[0,0], units='pix',
                                   sf=1./cls.angle_to_pix(1./cls.experiment['spatial-freq'][index]),
                                   ori=cls.experiment['angle'][index],
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index]))]

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        xrot = compute_xrot(cls.x, cls.z, cls.experiment['angle'][episode])
        return compute_grating(xrot,
                               spatial_freq=cls.experiment['spatial-freq'][episode],
                               contrast=cls.experiment['contrast'][episode])
        

            
class drifting_full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol,
                                ['spatial-freq', 'angle', 'contrast', 'speed'],
                                run_type='drifting')

    def get_patterns(self, index, parent=None):
        cls = (parent if parent is not None else self)
        return [visual.GratingStim(win=cls.win,
                                   size=10000, pos=[0,0], units='pix',
                                   sf=1./cls.angle_to_pix(1./cls.experiment['spatial-freq'][index]),
                                   ori=cls.experiment['angle'][index],
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index]))]
    
    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        xrot = compute_xrot(cls.x, cls.z, cls.experiment['angle'][episode])
        return compute_grating(xrot,
                               spatial_freq=cls.experiment['spatial-freq'][episode],
                               contrast=cls.experiment['contrast'][episode],
                               time_phase=cls.experiment['speed'][episode]*time_from_episode_start)

    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, enhance=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red', 'center':[0,0]}):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode, ax=ax,
                             parent=parent,
                             label=label,
                             enhance=enhance)

        arrow['direction'] = cls.experiment['angle'][episode]
        self.add_arrow(arrow, ax)

        return ax
    

        
#####################################################
##  ----    PRESENTING CENTERED GRATINGS    --- #####           
#####################################################

class center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol,
                                ['bg-color', 'x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast'],
                                run_type='static')

    def get_patterns(self, index, parent=None):
        cls = (parent if parent is not None else self)
        return [visual.GratingStim(win=cls.win,
                                   pos=[cls.angle_to_pix(cls.experiment['x-center'][index]),
                                        cls.angle_to_pix(cls.experiment['y-center'][index])],
                                   sf=1./cls.angle_to_pix(1./cls.experiment['spatial-freq'][index]),
                                   size= 2*cls.angle_to_pix(cls.experiment['radius'][index]),
                                   ori=cls.experiment['angle'][index],
                                   units='pix', mask='circle',
                                   # color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index]),
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index]))]
    
    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        xrot = compute_xrot(cls.x, cls.z, cls.experiment['angle'][episode],
                            xcenter=cls.experiment['x-center'][episode],
                            zcenter=cls.experiment['y-center'][episode])
        mask = (((cls.x-cls.experiment['x-center'][episode])**2+\
                 (cls.z-cls.experiment['y-center'][episode])**2)<=cls.experiment['radius'][episode]**2) # circle mask
        img = (1+cls.experiment['bg-color'][episode])/2.+0*cls.x
        img[mask] = compute_grating(xrot[mask], 
                                    contrast=cls.experiment['contrast'][episode],
                                    spatial_freq=cls.experiment['spatial-freq'][episode])
        return img




class drifting_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol,
                                ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'speed', 'bg-color'],
                                run_type='drifting')

    def get_patterns(self, index, parent=None):
        cls = (parent if parent is not None else self)
        return [visual.GratingStim(win=cls.win,
                                   pos=[cls.angle_to_pix(cls.experiment['x-center'][index]),
                                        cls.angle_to_pix(cls.experiment['y-center'][index])],
                                   sf=1./cls.angle_to_pix(1./cls.experiment['spatial-freq'][index]),
                                   size= 2*cls.angle_to_pix(cls.experiment['radius'][index]),
                                   ori=cls.experiment['angle'][index],
                                   units='pix', mask='circle',
                                   # color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index]),
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index]))]

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        """
        Need to implement it 
        """
        cls = (parent if parent is not None else self)
        xrot = compute_xrot(cls.x, cls.z, cls.experiment['angle'][episode],
                            xcenter=cls.experiment['x-center'][episode],
                            zcenter=cls.experiment['y-center'][episode])
        mask = (((cls.x-cls.experiment['x-center'][episode])**2+\
                 (cls.z-cls.experiment['y-center'][episode])**2)<=cls.experiment['radius'][episode]**2) # circle mask
        img = (1+cls.experiment['bg-color'][episode])/2.+0*cls.x
        img[mask] = compute_grating(xrot[mask], 
                                    contrast=cls.experiment['contrast'][episode],
                                    spatial_freq=cls.experiment['spatial-freq'][episode],
                                    time_phase=cls.experiment['speed'][episode]*time_from_episode_start)
        return img


    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, enhance=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red', 'center':[0,0]}):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode, ax=ax,
                             parent=parent,
                             label=label,
                             enhance=enhance)

        arrow['center'] = [cls.experiment['x-center'][episode],
                           cls.experiment['y-center'][episode]]
        arrow['direction'] = cls.experiment['angle'][episode]
        self.add_arrow(arrow, ax)

        return ax
    

#####################################################
##  ----    PRESENTING OFF-CENTERED GRATINGS    --- #####           
#####################################################

class off_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol,
                                ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color'],
                                run_type='static')

        
    def get_patterns(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return [visual.GratingStim(win=cls.win,
                                   size=10000, pos=[0,0], units='pix',
                                   sf=1./cls.angle_to_pix(1./cls.experiment['spatial-freq'][index]),
                                   ori=cls.experiment['angle'][index],
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index])),
                visual.GratingStim(win=cls.win,
                                   pos=[cls.angle_to_pix(cls.experiment['x-center'][index]),
                                        cls.angle_to_pix(cls.experiment['y-center'][index])],
                                   sf=1./cls.angle_to_pix(1./cls.experiment['spatial-freq'][index]),
                                   size= 2*cls.angle_to_pix(cls.experiment['radius'][index]),
                                   mask='circle', units='pix',
                                   color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index]))]


    def get_image(self, episode, time_from_episode_start=0, parent=None):
        """
        Need to implement it 
        """
        cls = (parent if parent is not None else self)
        xrot = compute_xrot(cls.x, cls.z, cls.experiment['angle'][episode],
                            xcenter=cls.experiment['x-center'][episode],
                            zcenter=cls.experiment['y-center'][episode])
        mask = (((cls.x-cls.experiment['x-center'][episode])**2+\
                 (cls.z-cls.experiment['y-center'][episode])**2)<=cls.experiment['radius'][episode]**2) # circle mask
        img = compute_grating(xrot, 
                              contrast=cls.experiment['contrast'][episode],
                              spatial_freq=cls.experiment['spatial-freq'][episode])
        img[mask] = (1+cls.experiment['bg-color'][episode])/2.+0*cls.x[mask]
        return img
    
    
class drifting_off_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color', 'speed'], run_type='drifting')

    def get_patterns(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return [\
                # Surround grating
                visual.GratingStim(win=cls.win,
                                   size=10000, pos=[0,0], units='pix',
                                   sf=1./cls.angle_to_pix(1./cls.experiment['spatial-freq'][index]),
                                   ori=cls.experiment['angle'][index],
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index])),
                # + center Mask
                visual.GratingStim(win=cls.win,
                                   pos=[cls.angle_to_pix(cls.experiment['x-center'][index]),
                                        cls.angle_to_pix(cls.experiment['y-center'][index])],
                                   sf=1./cls.angle_to_pix(1./cls.experiment['spatial-freq'][index]),
                                   size= 2*cls.angle_to_pix(cls.experiment['radius'][index]),
                                   mask='circle', units='pix',
                                   color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index]))]


    def get_image(self, episode, time_from_episode_start=0, parent=None):
        """
        Need to implement it 
        """
        cls = (parent if parent is not None else self)
        xrot = compute_xrot(cls.x, cls.z, cls.experiment['angle'][episode],
                            xcenter=cls.experiment['x-center'][episode],
                            zcenter=cls.experiment['y-center'][episode])
        mask = (((cls.x-cls.experiment['x-center'][episode])**2+\
                 (cls.z-cls.experiment['y-center'][episode])**2)<=cls.experiment['radius'][episode]**2) # circle mask
        img = compute_grating(xrot, 
                              contrast=cls.experiment['contrast'][episode],
                              spatial_freq=cls.experiment['spatial-freq'][episode],
                              time_phase=cls.experiment['speed'][episode]*time_from_episode_start)
        img[mask] = (1+cls.experiment['bg-color'][episode])/2.+0*cls.x[mask]
        return img
    


    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, enhance=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red', 'center':[0,0]}):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode, ax=ax,
                             parent=parent,
                             label=label,
                             enhance=enhance)

        arrow['center'] = [cls.experiment['x-center'][episode],
                           cls.experiment['y-center'][episode]]
        arrow['direction'] = cls.experiment['angle'][episode]
        self.add_arrow(arrow, ax)

        return ax
    
#####################################################
##  ----    PRESENTING SURROUND GRATINGS    --- #####           
#####################################################
        
class surround_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius-start', 'radius-end','spatial-freq', 'angle', 'contrast', 'bg-color'], run_type='static')

    def get_patterns(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return [visual.GratingStim(win=cls.win,
                                   size=1000, pos=[0,0], sf=0,
                                   color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index])),
                visual.GratingStim(win=cls.win,
                                   pos=[cls.experiment['x-center'][index], cls.experiment['y-center'][index]],
                                   size=cls.experiment['radius-end'][index],
                                   mask='circle', 
                                   sf=cls.experiment['spatial-freq'][index],
                                   ori=cls.experiment['angle'][index],
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index])),
                visual.GratingStim(win=cls.win,
                                   pos=[cls.experiment['x-center'][index], cls.experiment['y-center'][index]],
                                   size=cls.experiment['radius-start'][index],
                                   mask='circle', sf=0,
                                   color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index]))]
    
    def get_image(self, episode, time_from_episode_start=0, parent=None):
        """
        Need to implement it 
        """
        cls = (parent if parent is not None else self)
        xrot = compute_xrot(cls.x, cls.z, cls.experiment['angle'][episode],
                            xcenter=cls.experiment['x-center'][episode],
                            zcenter=cls.experiment['y-center'][episode])
        mask1 = (((cls.x-cls.experiment['x-center'][episode])**2+\
                  (cls.z-cls.experiment['y-center'][episode])**2)>=cls.experiment['radius-start'][episode]**2) # circle mask
        mask2 = (((cls.x-cls.experiment['x-center'][episode])**2+\
                  (cls.z-cls.experiment['y-center'][episode])**2)<=cls.experiment['radius-end'][episode]**2) # circle mask
        img = (1+cls.experiment['bg-color'][episode])/2.+0*cls.x
        img[mask1 & mask2] = compute_grating(xrot[mask1 & mask2], 
                                             contrast=cls.experiment['contrast'][episode],
                                             spatial_freq=cls.experiment['spatial-freq'][episode])
        return img

    
class drifting_surround_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius-start', 'radius-end','spatial-freq', 'angle', 'contrast', 'bg-color', 'speed'], run_type='drifting')

    def get_patterns(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return [visual.GratingStim(win=cls.win,
                                   size=1000, pos=[0,0], sf=0,contrast=0,
                                   color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index])),
                visual.GratingStim(win=cls.win,
                                   pos=[cls.experiment['x-center'][index], cls.experiment['y-center'][index]],
                                   size=cls.experiment['radius-end'][index],
                                   mask='circle', 
                                   sf=cls.experiment['spatial-freq'][index],
                                   ori=cls.experiment['angle'][index],
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index])),
                visual.GratingStim(win=cls.win,
                                   pos=[cls.experiment['x-center'][index], cls.experiment['y-center'][index]],
                                   size=cls.experiment['radius-start'][index],
                                   mask='circle', sf=0,contrast=0,
                                   color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index]))]

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        """
        Need to implement it 
        """
        cls = (parent if parent is not None else self)
        xrot = compute_xrot(cls.x, cls.z, cls.experiment['angle'][episode],
                            xcenter=cls.experiment['x-center'][episode],
                            zcenter=cls.experiment['y-center'][episode])
        mask1 = (((cls.x-cls.experiment['x-center'][episode])**2+\
                  (cls.z-cls.experiment['y-center'][episode])**2)>=cls.experiment['radius-start'][episode]**2) # circle mask
        mask2 = (((cls.x-cls.experiment['x-center'][episode])**2+\
                  (cls.z-cls.experiment['y-center'][episode])**2)<=cls.experiment['radius-end'][episode]**2) # circle mask
        img = (1+cls.experiment['bg-color'][episode])/2.+0*cls.x
        img[mask1 & mask2] = compute_grating(xrot[mask1 & mask2], 
                                             contrast=cls.experiment['contrast'][episode],
                                             spatial_freq=cls.experiment['spatial-freq'][episode],
                                             time_phase=cls.experiment['speed'][episode]*time_from_episode_start)
        return img
    

    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, enhance=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red', 'center':[0,0]}):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode, ax=ax,
                             parent=parent,
                             label=label,
                             enhance=enhance)

        arrow['direction'] = cls.experiment['angle'][episode]
        self.add_arrow(arrow, ax)

        return ax
    
    
#####################################################
##  -- PRESENTING APPEARING GAUSSIAN BLOBS  --  #####           
#####################################################

class gaussian_blobs(visual_stim):
    
    def __init__(self, protocol):

        super().__init__(protocol)
        
        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 30.
        if 'appearance_threshold' not in protocol:
            protocol['appearance_threshold'] = 2.5 # 
        self.frame_refresh = protocol['movie_refresh_freq']
        
        super().init_experiment(self.protocol,
                                ['x-center', 'y-center', 'radius','center-time', 'extent-time', 'contrast', 'bg-color'],
                                run_type='images_sequence')
        
            
    def get_frames_sequence(self, index, parent=None):
        """
        Generator creating a random number of chunks (but at most max_chunks) of length chunk_length containing
        random samples of sin([0, 2pi]).
        """
        cls = (parent if parent is not None else self)
        bg = np.ones(cls.screen['resolution'])*cls.experiment['bg-color'][index]
        interval = cls.experiment['time_stop'][index]-cls.experiment['time_start'][index]

        contrast = cls.experiment['contrast'][index]
        xcenter, zcenter = cls.experiment['x-center'][index], cls.experiment['y-center'][index]
        radius = cls.experiment['radius'][index]
        bg_color = cls.experiment['bg-color'][index]
        
        t0, sT = cls.experiment['center-time'][index], cls.experiment['extent-time'][index]
        itstart = np.max([0, int((t0-cls.protocol['appearance_threshold']*sT)*cls.protocol['movie_refresh_freq'])])
        itend = np.min([int(interval*cls.protocol['movie_refresh_freq']),
                        int((t0+cls.protocol['appearance_threshold']*sT)*cls.protocol['movie_refresh_freq'])])

        times, FRAMES = np.zeros(int(1.2*interval*cls.protocol['movie_refresh_freq']), dtype=int), []
        # the pre-time
        FRAMES.append(2*bg_color-1.+0.*self.x)
        times[:itstart] = 0
        for iframe, it in enumerate(np.arange(itstart, itend)):
            img = 2*(np.exp(-((self.x-xcenter)**2+(self.z-zcenter)**2)/2./radius**2)*\
                     contrast*np.exp(-(it/cls.protocol['movie_refresh_freq']-t0)**2/2./sT**2)+bg_color)-1.
            FRAMES.append(img)
            times[it] = iframe
        # the post-time
        FRAMES.append(2*bg_color-1.+0.*self.x)
        times[itend:] = len(FRAMES)-1
            
        return times, FRAMES

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        xcenter, zcenter = cls.experiment['x-center'][episode], cls.experiment['y-center'][episode]
        radius = cls.experiment['radius'][episode]
        t0, sT = cls.experiment['center-time'][episode], cls.experiment['extent-time'][episode]
        return np.exp(-((cls.x-xcenter)**2+(cls.z-zcenter)**2)/2./radius**2)*\
            np.exp(-(time_from_episode_start-t0)**2/2./sT**2)*\
            cls.experiment['contrast'][episode]+cls.experiment['bg-color'][episode]
    

#####################################################
##  ----    PRESENTING NATURAL IMAGES       --- #####
#####################################################

NI_directory = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'NI_bank')
        
class natural_image(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['Image-ID'], run_type='image')
        self.NIarray = []
        for filename in os.listdir(NI_directory):
            img = load(os.path.join(NI_directory, filename))
            new_img = adapt_to_screen_resolution(img, self.screen)
            self.NIarray.append(2*img_after_hist_normalization(new_img)-1.)

    def get_frame(self, index, parent=None):
        cls = (parent if parent is not None else self)
        return cls.NIarray[int(cls.experiment['Image-ID'][index])]
                       

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        return cls.NIarray[int(cls.experiment['Image-ID'][index])]
            
#####################################################
##  --    WITH VIRTUAL SCENE EXPLORATION    --- #####
#####################################################


def generate_VSE(duration=5,
                 mean_saccade_duration=2.,# in s
                 std_saccade_duration=1.,# in s
                 saccade_amplitude=100, # in pixels, TO BE PUT IN DEGREES
                 seed=0):
    """
    to do: clean up the VSE generator
    """
    print('generating Virtual-Scene-Exploration [...]')
    
    np.random.seed(seed)
    
    tsaccades = np.cumsum(np.clip(np.abs(mean_saccade_duration+np.random.randn(int(1.5*duration/mean_saccade_duration))*std_saccade_duration),
                                  mean_saccade_duration/4., 1.75*mean_saccade_duration))

    x = np.random.uniform(1, 2*saccade_amplitude, size=len(tsaccades))
    y = np.random.uniform(1, 2*saccade_amplitude, size=len(tsaccades))
    
    # x = np.array(np.clip((np.random.randn(len(tsaccades))+1)*saccade_amplitude, 1, 2*saccade_amplitude), dtype=int)
    # y = np.array(np.clip((np.random.randn(len(tsaccades))+1)*saccade_amplitude, 1, 2*saccade_amplitude), dtype=int)
    
    return {'t':np.array([0]+list(tsaccades)),
            'x':np.array([0]+list(x)),
            'y':np.array([0]+list(y)),
            'max_amplitude':saccade_amplitude}

            

class natural_image_vse(visual_stim):

    
    def __init__(self, protocol):

        super().__init__(protocol)
        super().init_experiment(protocol, ['Image-ID', 'VSE-seed',
                                           'mean-saccade-duration', 'std-saccade-duration',
                                           'vary-VSE-with-Image', 'saccade-amplitude'],
                                run_type='images_sequence')

        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 30.
        self.frame_refresh = protocol['movie_refresh_freq']

        # initializing set of NI
        self.NIarray = []
        for filename in os.listdir(NI_directory):
            img = load(os.path.join(NI_directory, filename))
            new_img = adapt_to_screen_resolution(img, self.screen)
            self.NIarray.append(2*img_after_hist_normalization(new_img)-1.)


    def compute_shifted_image(self, img, ix, iy):
        """
        translate saccdes in degree in pixels here
        """
        sx, sy = img.shape
        new_im = np.zeros(img.shape)
	# print(img[:sx-ix,:sy-iy].shape)
        # print(img[sx-ix:,:].shape)
        # print(img[:,sy-iy:].shape)
        # print(img[sx-ix:,sy-iy:].shape)
        new_im[ix:,iy:] = img[:sx-ix,:sy-iy]
        new_im[:ix,:] = img[sx-ix:,:]
        new_im[:,:iy] = img[:,sy-iy:]
        new_im[:ix,:iy] = img[sx-ix:,sy-iy:]
        return new_im
    

    def get_seed(self, index, parent=None):
        cls = (parent if parent is not None else self)
        if cls.experiment['vary-VSE-with-Image'][index]==1:
            return int(cls.experiment['VSE-seed'][index]+1000*cls.experiment['Image-ID'][index])
        else:
            return int(cls.experiment['VSE-seed'][index])

        
    def get_frames_sequence(self, index, parent=None):
        cls = (parent if parent is not None else self)
        seed = self.get_seed(index, parent=cls)

        vse = generate_VSE(duration=cls.experiment['time_duration'][index],
                           mean_saccade_duration=cls.experiment['mean-saccade-duration'][index],
                           std_saccade_duration=cls.experiment['std-saccade-duration'][index],
                           saccade_amplitude=cls.angle_to_pix(cls.experiment['saccade-amplitude'][index]),
                           seed=seed)

        img = self.NIarray[int(cls.experiment['Image-ID'][index])]
            
        interval = cls.experiment['time_stop'][index]-cls.experiment['time_start'][index]
        times, FRAMES = np.zeros(int(1.2*interval*cls.protocol['movie_refresh_freq']), dtype=int), []
        Times = np.arange(int(1.2*interval*cls.protocol['movie_refresh_freq']))/cls.protocol['movie_refresh_freq']

        for i, t in enumerate(vse['t']):
            FRAMES.append(self.compute_shifted_image(img, int(vse['x'][i]), int(vse['y'][i])))
            times[Times>=t] = int(i)
            
        return times, FRAMES

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        return (1.+self.NIarray[int(cls.experiment['Image-ID'][episode])])/2.

    
#####################################################
##  ----    PRESENTING BINARY NOISE         --- #####
#####################################################

class sparse_noise(visual_stim):
    
    def __init__(self, protocol):

        super().__init__(protocol)

        self.noise_gen = sparse_noise_generator(duration=protocol['presentation-duration'],
                                                screen=self.screen,
                                                sparseness=protocol['sparseness (%)']/100.,
                                                square_size=protocol['square-size (deg)'],
                                                bg_color=protocol['bg-color (lum.)'],
                                                contrast=protocol['contrast (norm.)'],
                                                noise_mean_refresh_time=protocol['mean-refresh-time (s)'],
                                                noise_rdm_jitter_refresh_time=protocol['jitter-refresh-time (s)'],
                                                seed=protocol['noise-seed (#)'])

        self.experiment = {}
        self.experiment['index'] = np.arange(len(self.noise_gen.events)-1) 
        self.experiment['interstim'] = np.zeros(len(self.noise_gen.events)-1) 
        self.experiment['interstim-screen'] = np.zeros(len(self.noise_gen.events)-1) 
        self.experiment['time_start'] = self.noise_gen.events[:-1]+protocol['presentation-prestim-period']
        self.experiment['time_stop'] = self.noise_gen.events[1:]+protocol['presentation-prestim-period']
        self.experiment['frame_run_type'] = ['image' for i in self.experiment['index']]
        self.experiment['time_duration'] = self.experiment['time_stop']-self.experiment['time_start']

        
    def get_frame(self, index, parent=None):
        cls = (parent if parent is not None else self)
        return self.noise_gen.get_frame(index).T

    def get_image(self, index, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        return (1+self.noise_gen.get_frame(index).T)/2.
    

class dense_noise(visual_stim):

    def __init__(self, protocol):

        super().__init__(protocol)

        self.noise_gen = dense_noise_generator(duration=protocol['presentation-duration'],
                                                screen=self.screen,
                                                square_size=protocol['square-size (deg)'],
                                                contrast=protocol['contrast (norm.)'],
                                                noise_mean_refresh_time=protocol['mean-refresh-time (s)'],
                                                noise_rdm_jitter_refresh_time=protocol['jitter-refresh-time (s)'],
                                                seed=protocol['noise-seed (#)'])

        self.experiment = {}
        self.experiment['index'] = np.arange(len(self.noise_gen.events))
        self.experiment['interstim'] = np.zeros(len(self.noise_gen.events)-1)
        self.experiment['interstim-screen'] = np.zeros(len(self.noise_gen.events)-1)
        self.experiment['time_start'] = self.noise_gen.events[:-1]+protocol['presentation-prestim-period']
        self.experiment['time_stop'] = self.noise_gen.events[1:]+protocol['presentation-prestim-period']
        self.experiment['frame_run_type'] = ['image' for i in self.experiment['index']]
        self.experiment['time_duration'] = self.experiment['time_stop']-self.experiment['time_start']
        
    def get_frame(self, index, parent=None):
        cls = (parent if parent is not None else self)
        return self.noise_gen.get_frame(index).T

    def get_image(self, index, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        return (1+self.noise_gen.get_frame(index).T)/2.
            

#####################################################
##  ----    PRESENTING MOVING DOTS          --- #####
#####################################################

class line_moving_dots(visual_stim):

    def __init__(self, protocol):

        super().__init__(protocol)

        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 30.
        self.frame_refresh = protocol['movie_refresh_freq']
        
        super().init_experiment(protocol,
                                ['speed', 'bg-color', 'ndots', 'spacing',
                                 'direction', 'size', 'dotcolor'],
                                run_type='images_sequence')

    def add_dot(self, image, pos, size, color, type='square'):
        """
        add dot
        """
        if type=='square':
            cond = (self.x>(pos[0]-size/2)) & (self.x<(pos[0]+size/2)) & (self.z>(pos[1]-size/2)) & (self.z<(pos[1]+size/2))
        else:
            cond = np.sqrt((self.x-pos[0])**2+(self.z-pos[1])**2)<size
        image[cond] = color


    def get_starting_point_and_direction(self, index, cls):
        
        interval = cls.experiment['time_stop'][index]-cls.experiment['time_start'][index]
        
        line = np.arange(int(cls.experiment['ndots'][index]))*cls.experiment['spacing'][index]
        X0, Y0 = [], []
        if cls.experiment['direction'][index]==0:
            # right -> left
            dx_per_time, dy_per_time = -cls.experiment['speed'][index], 0
            X0 = np.zeros(int(cls.experiment['ndots'][index]))-interval*dx_per_time/2.
            Y0 = line-line.mean()
        elif cls.experiment['direction'][index]==180:
            # left -> right
            dx_per_time, dy_per_time = cls.experiment['speed'][index], 0
            X0 = np.zeros(int(cls.experiment['ndots'][index]))-interval*dx_per_time/2.
            Y0 = line-line.mean()
        elif cls.experiment['direction'][index]==90:
            # top -> bottom
            dx_per_time, dy_per_time = 0, -cls.experiment['speed'][index]
            Y0 = np.zeros(int(cls.experiment['ndots'][index]))-interval*dy_per_time/2.
            X0 = line-line.mean()
        elif cls.experiment['direction'][index]==270:
            # top -> bottom
            dx_per_time, dy_per_time = 0, cls.experiment['speed'][index]
            Y0 = np.zeros(int(cls.experiment['ndots'][index]))-interval*dy_per_time/2.
            X0 = line-line.mean()
        else:
            print('direction "%i" not implemented !' % cls.experiment['direction'][index])

        return X0, Y0, dx_per_time, dy_per_time
        
    def get_frames_sequence(self, index, parent=None):
        """
        
        """
        cls = (parent if parent is not None else self)
        bg = np.ones(cls.screen['resolution'])*cls.experiment['bg-color'][index]

        interval = cls.experiment['time_stop'][index]-cls.experiment['time_start'][index]
        X0, Y0, dx_per_time, dy_per_time = self.get_starting_point_and_direction(index, cls)

        bg_color = cls.experiment['bg-color'][index]
        
        itstart, itend = 0, int(1.2*interval*cls.protocol['movie_refresh_freq'])

        times, FRAMES = [], []
        for iframe, it in enumerate(np.arange(itend)):
            time = it/cls.protocol['movie_refresh_freq']
            img = 2*bg_color-1.+0.*self.x
            for x0, y0 in zip(X0, Y0):
                # adding the dots one by one
                new_position = (x0+dx_per_time*time, y0+dy_per_time*time)
                self.add_dot(img, new_position,
                             cls.experiment['size'][index],
                             cls.experiment['dotcolor'][index])
            FRAMES.append(img)
            times.append(iframe)
            
        return times, FRAMES


    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        img = 0*cls.x+cls.experiment['bg-color'][episode]
        X0, Y0, dx_per_time, dy_per_time = self.get_starting_point_and_direction(episode, cls)
        for x0, y0 in zip(X0, Y0):
            new_position = (x0+dx_per_time*time_from_episode_start,
                            y0+dy_per_time*time_from_episode_start)
            self.add_dot(img, new_position,
                         cls.experiment['size'][episode],
                         cls.experiment['dotcolor'][episode])
        return img

    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, enhance=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        cls = (parent if parent is not None else self)
        tcenter_minus = .43*(cls.experiment['time_stop'][episode]-\
                             cls.experiment['time_start'][episode])
        ax = self.show_frame(episode, ax=ax, label=label, enhance=enhance,
                             time_from_episode_start=tcenter_minus,
                             parent=parent)

        direction = cls.experiment['direction'][episode]

        # print(direction)
        arrow['direction'] = ((direction+180)%180)+180
        # print(arrow['direction'])
        
        for shift in [-.5, 0, .5]:
            arrow['center'] = [shift*np.sin(np.pi/180.*direction)*cls.screen['width'],
                               shift*np.cos(np.pi/180.*direction)*cls.screen['height']]
            self.add_arrow(arrow, ax)

        return ax
    
        
class random_dots(visual_stim):
    """
    draft, to be improved
    """
    def __init__(self, protocol):

        super().__init__(protocol)
        super().init_experiment(protocol,
                                ['speed'],
                                run_type='static')

    def get_patterns(self, index, parent=None):
        cls = (parent if parent is not None else self)
        return [visual.DotStim(win=cls.win,
                               dotSize=cls.angle_to_pix(10),
                               fieldPos=(0, 0),#cls.angle_to_pix(70)),
                               fieldSize=cls.win.size,
                               speed=3.,
                               dotLife=-1,
                               units='pix',
                               nDots=10,
                               color=-1,
                               signalDots='same',
                               noiseDots='walk',
                               dir=0,
                               coherence=1.)]
    
    def get_frame(self, index, parent=None):
        cls = (parent if parent is not None else self)
        return self.noise_gen.get_frame(index).T

    def get_image(self, index, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        return (1+self.noise_gen.get_frame(index).T)/2.

#####################################################
##  ----    PRESENTING LOOMING STIM         --- #####
#####################################################

class looming_stim(visual_stim):

    def __init__(self, protocol):

        super().__init__(protocol)

        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 30.
        self.frame_refresh = protocol['movie_refresh_freq']
        
        super().init_experiment(protocol,
                                ['radius-start', 'radius-end',
                                 'x-center', 'y-center',
                                 'color', 'looming-nonlinearity', 'looming-duration',
                                 'end-duration', 'bg-color'],
                                run_type='images_sequence')

    def add_dot(self, image, pos, size, color):
        """
        add dot
        """
        cond = np.sqrt((self.x-pos[0])**2+(self.z-pos[1])**2)<size
        image[cond] = color

    # def compute_looming_trajectory(self, size_to_speed_ratio, dt=30e-3, start_size=0.5, end_size=100):
    #     """
    #     from:
    #     Computation of Object Approach by a Wide-Field, Motion-Sensitive Neuron
    #     Fabrizio Gabbiani, Holger G. Krapp and Gilles Laurent
    #     Journal of Neuroscience 1 February 1999, 19 (3) 1122-1141; DOI: https://doi.org/10.1523/JNEUROSCI.19-03-01122.1999
    #
    #     TO BE FIXED     
    #     """
    #     times, angles = [0], [end_size*np.pi/180.]
    #     i=0
    #     while angles[-1]>(start_size*np.pi/180.) and i<200:
    #         times.append(times[-1]-dt)
    #         angles.append(angles[-1]-2*dt*size_to_speed_ratio/(times[-1]**2+size_to_speed_ratio**2))
    #         i+=1
    #     return np.array(times)-times[-1], np.array(angles)[::-1]*180/np.pi
    
    def compute_looming_trajectory(self, duration=1., nonlinearity=2, dt=30e-3, start_size=0.5, end_size=100):
        """ SIMPLE """
        times = np.arange(int(duration/dt))*dt
        angles = np.linspace(0, 1, len(times))**nonlinearity
        return times, angles*(end_size-start_size)+start_size
    
        
    def get_frames_sequence(self, index, parent=None):
        """
        
        """
        cls = (parent if parent is not None else self)
        
        interval = cls.experiment['time_stop'][index]-cls.experiment['time_start'][index]

        # background frame:
        bg = 2*cls.experiment['bg-color'][index]-1.+0.*self.x

        t, angles = self.compute_looming_trajectory(duration=cls.experiment['looming-duration'][index],
                                                    nonlinearity=cls.experiment['looming-nonlinearity'][index],
                                                    dt=1./self.frame_refresh,
                                                    start_size=cls.experiment['radius-start'][index],
                                                    end_size=cls.experiment['radius-end'][index])

        itend = int(1.2*interval*cls.protocol['movie_refresh_freq'])

        times_index_to_frames, FRAMES = [0], [bg.copy()]
        for it in range(len(t))[1:]:
            img = bg.copy()
            self.add_dot(img, (cls.experiment['x-center'][index], cls.experiment['y-center'][index]),
                         angles[it],
                         cls.experiment['color'][index])
            FRAMES.append(img)
            times_index_to_frames.append(it)
        it = len(t)-1
        while it<len(t)+int(cls.experiment['end-duration'][index]*cls.protocol['movie_refresh_freq']):
            times_index_to_frames.append(len(FRAMES)-1) # the last one
            it+=1
        while it<itend:
            times_index_to_frames.append(0) # the first one (bg)
            it+=1

        return times_index_to_frames, FRAMES

    def get_image(self, index, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        img = cls.experiment['bg-color'][index]+0.*self.x
        self.add_dot(img, (cls.experiment['x-center'][index],
                           cls.experiment['y-center'][index]),
                     cls.experiment['radius-end'][index]/4.,
                     cls.experiment['color'][index])
        return img

    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, enhance=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode, ax=ax, label=label, enhance=enhance,
                             parent=parent)

        l = cls.experiment['radius-end'][episode]/3.8 # just like above
        for d in np.linspace(0, 2*np.pi, 3, endpoint=False):
            arrow['center'] = [cls.experiment['x-center'][episode]+np.cos(d)*l+\
                               np.cos(d)*arrow['length']/2.,
                               cls.experiment['y-center'][episode]+np.sin(d)*l+\
                               np.sin(d)*arrow['length']/2.]
                
            arrow['direction'] = -180*d/np.pi
            self.add_arrow(arrow, ax)
            
        return ax
    
class moving_dots_static_patch(visual_stim):

    def __init__(self, protocol):

        super().__init__(protocol)

        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 30.
        self.frame_refresh = protocol['movie_refresh_freq']
        
        super().init_experiment(protocol,
                                ['speed', 'bg-color', 'ndots', 'spacing',
                                 'direction', 'size', 'dotcolor', 'patch-delay',
                                 'patch-radius', 'patch-contrast', 'patch-spatial-freq', 'patch-angle'],
                                run_type='images_sequence')


    def add_patch(self, image,
                  angle=0,
                  radius=10,
                  spatial_freq=0.1,
                  contrast=1.,
                  time_phase=0.,
                  xcenter=0, zcenter=0):

        xrot = compute_xrot(self.x, self.z, angle,
                            xcenter=xcenter, zcenter=zcenter)

        cond = ((self.x-xcenter)**2+(self.z-zcenter)**2)<radius**2
        image[cond] = 2*compute_grating(xrot[cond],
                                      spatial_freq=spatial_freq,
                                      contrast=contrast,
                                      time_phase=time_phase)-1

        
    def add_dot(self, image, pos, size, color, type='square'):
        """
        add dot
        """
        if type=='square':
            cond = (self.x>(pos[0]-size/2)) & (self.x<(pos[0]+size/2)) & (self.z>(pos[1]-size/2)) & (self.z<(pos[1]+size/2))
        else:
            cond = np.sqrt((self.x-pos[0])**2+(self.z-pos[1])**2)<size
        image[cond] = color


    def get_starting_point_and_direction(self, index, cls):
        
        interval = cls.experiment['time_stop'][index]-cls.experiment['time_start'][index]
        
        line = np.arange(int(cls.experiment['ndots'][index]))*cls.experiment['spacing'][index]
        X0, Y0 = [], []
        if cls.experiment['direction'][index]==0:
            # right -> left
            dx_per_time, dy_per_time = -cls.experiment['speed'][index], 0
            X0 = np.zeros(int(cls.experiment['ndots'][index]))-interval*dx_per_time/2.
            Y0 = line-line.mean()
        elif cls.experiment['direction'][index]==180:
            # left -> right
            dx_per_time, dy_per_time = cls.experiment['speed'][index], 0
            X0 = np.zeros(int(cls.experiment['ndots'][index]))-interval*dx_per_time/2.
            Y0 = line-line.mean()
        elif cls.experiment['direction'][index]==90:
            # top -> bottom
            dx_per_time, dy_per_time = 0, -cls.experiment['speed'][index]
            Y0 = np.zeros(int(cls.experiment['ndots'][index]))-interval*dy_per_time/2.
            X0 = line-line.mean()
        elif cls.experiment['direction'][index]==270:
            # top -> bottom
            dx_per_time, dy_per_time = 0, cls.experiment['speed'][index]
            Y0 = np.zeros(int(cls.experiment['ndots'][index]))-interval*dy_per_time/2.
            X0 = line-line.mean()
        else:
            print('direction "%i" not implemented !' % cls.experiment['direction'][index])

        return X0, Y0, dx_per_time, dy_per_time
        
    def get_frames_sequence(self, index, parent=None):
        """
        
        """
        cls = (parent if parent is not None else self)
        bg = np.ones(cls.screen['resolution'])*cls.experiment['bg-color'][index]

        interval = cls.experiment['time_stop'][index]-cls.experiment['time_start'][index]
        X0, Y0, dx_per_time, dy_per_time = self.get_starting_point_and_direction(index, cls)

        bg_color = cls.experiment['bg-color'][index]
        
        itstart, itend = 0, int(1.2*interval*cls.protocol['movie_refresh_freq'])

        times, FRAMES = [], []
        for iframe, it in enumerate(np.arange(itend)):
            time = it/cls.protocol['movie_refresh_freq']
            img = 2*bg_color-1.+0.*self.x
            for x0, y0 in zip(X0, Y0):
                # adding the dots one by one
                new_position = (x0+dx_per_time*time, y0+dy_per_time*time)
                self.add_dot(img, new_position,
                             cls.experiment['size'][index],
                             cls.experiment['dotcolor'][index])
            if time>cls.experiment['patch-delay'][index]:
                self.add_patch(img,
                               angle=cls.experiment['patch-angle'][index],
                               radius=cls.experiment['patch-radius'][index],
                               spatial_freq=cls.experiment['patch-spatial-freq'][index],
                               contrast=cls.experiment['patch-contrast'][index])
                
            FRAMES.append(img)
            times.append(iframe)
            
        return times, FRAMES


    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        img = 0*cls.x+cls.experiment['bg-color'][episode]
        X0, Y0, dx_per_time, dy_per_time = self.get_starting_point_and_direction(episode, cls)
        for x0, y0 in zip(X0, Y0):
            new_position = (x0+dx_per_time*time_from_episode_start,
                            y0+dy_per_time*time_from_episode_start)
            self.add_dot(img, new_position,
                         cls.experiment['size'][episode],
                         cls.experiment['dotcolor'][episode])
        return img

    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, enhance=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        cls = (parent if parent is not None else self)
        tcenter_minus = .43*(cls.experiment['time_stop'][episode]-\
                             cls.experiment['time_start'][episode])
        ax = self.show_frame(episode, ax=ax, label=label, enhance=enhance,
                             time_from_episode_start=tcenter_minus,
                             parent=parent)

        direction = cls.experiment['direction'][episode]

        # print(direction)
        arrow['direction'] = ((direction+180)%180)+180
        # print(arrow['direction'])
        
        for shift in [-.5, 0, .5]:
            arrow['center'] = [shift*np.sin(np.pi/180.*direction)*cls.screen['width'],
                               shift*np.cos(np.pi/180.*direction)*cls.screen['height']]
            self.add_arrow(arrow, ax)

        return ax
    
    
if __name__=='__main__':

    import json, tempfile
    from pathlib import Path
    
    # with open('physion/exp/protocols/CB1-project-protocol.json', 'r') as fp:
    # with open('physion/exp/protocols/ff-drifting-grating-contrast-curve-log-spaced.json', 'r') as fp:
    # with open('physion/intrinsic/vis_stim/up.json', 'r') as fp:
    # with open('physion/exp/protocols/looming-stim.json', 'r') as fp:
    with open('physion/exp/protocols/mixed-moving-dots-static-patch.json', 'r') as fp:
        protocol = json.load(fp)

    protocol['demo'] = True
    
    class df:
        def __init__(self):
            pass
        def get(self):
            return tempfile.gettempdir()
        
    class dummy_parent:
        def __init__(self):
            self.stop_flag = False
            self.datafolder = df()

    stim = build_stim(protocol)
    parent = dummy_parent()
    stim.run(parent)
    stim.close()

