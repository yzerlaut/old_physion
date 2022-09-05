import numpy as np
import itertools, os, sys, pathlib, time, json, tempfile
try:
    from psychopy import visual, core, event, clock, monitors
except ModuleNotFoundError:
    pass

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from screens import SCREENS
from preprocess_NI import load, img_after_hist_normalization, adapt_to_screen_resolution


def build_stim(protocol):
    """
    """

    if (protocol['Presentation']=='multiprotocol'):
        return multiprotocol(protocol)
    else:
        protocol_name = protocol['Stimulus'].replace('-image','').replace('-', '_').replace('+', '_')
        if hasattr(sys.modules[__name__], protocol_name):
            return getattr(sys.modules[__name__], protocol_name)(protocol) # e.g. returns "center_grating_image_stim(protocol)"
        else:
            print(protocol_name)
            print(protocol)
            print('\n /!\ Protocol not recognized ! /!\ \n ')
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

    def __init__(self,
                 protocol,
                 demo=False):
        """
        """
        self.protocol = protocol
        self.screen = SCREENS[self.protocol['Screen']]
        self.buffer = None # by default, non-buffered data
        self.buffer_delay = 0            

        self.protocol['movie_refresh_freq'] = protocol['movie_refresh_freq'] if 'movie_refresh_freq' in protocol else 10.

        if demo or (('demo' in self.protocol) and self.protocol['demo']):
            # --------------------- #
            ## ---- DEMO MODE ---- ##    we override the parameters
            # --------------------- #
            self.screen['monitoring_square']['size'] = int(600*self.screen['monitoring_square']['size']/self.screen['resolution'][0])
            self.screen['resolution'] = (800,int(800*self.screen['resolution'][1]/self.screen['resolution'][0]))
            self.screen['screen_id'] = 0
            self.screen['fullscreen'] = False

        # then we can initialize the angle
        self.set_angle_meshgrid()

        if not ('no-window' in self.protocol):

            self.monitor = monitors.Monitor(self.screen['name'])
            self.monitor.setDistance(self.screen['distance_from_eye'])
            self.k, self.gamma = self.screen['gamma_correction']['k'], self.screen['gamma_correction']['gamma']

            self.win = visual.Window(self.screen['resolution'], monitor=self.monitor,
                                     screen=self.screen['screen_id'], fullscr=self.screen['fullscreen'],
                                     units='pix',
                                     color=self.gamma_corrected_lum(self.protocol['presentation-prestim-screen']))

            # ---- blank screens ----

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


            # ---- monitoring square properties ----

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

    def cm_to_angle(self, value):
        return 180./np.pi*np.arctan(value/self.screen['distance_from_eye'])

    def pix_to_angle(self, value):
        return self.cm_to_angle(value/self.screen['resolution'][0]*self.screen['width'])

    def set_angle_meshgrid(self):
        """
        #  ------- for simplicity -------  #
        # we linearize the arctan function #
        """
        dAngle_per_pix = self.pix_to_angle(1.)
        x, z = np.meshgrid(dAngle_per_pix*(np.arange(self.screen['resolution'][0])-self.screen['resolution'][0]/2.),
                           dAngle_per_pix*(np.arange(self.screen['resolution'][1])-self.screen['resolution'][1]/2.),
                           indexing='xy')
        self.x, self.z = x.T, z.T

    def angle_to_pix(self, angle):
        # using the above linear approx, the relationship is just the inverse:
        return angle/self.pix_to_angle(1.)

    # some general grating functions
    def compute_rotated_coords(self, angle,
                               xcenter=0, zcenter=0):
        return (self.x-xcenter)*np.cos(angle/180.*np.pi)+(self.z-zcenter)*np.sin(angle/180.*np.pi)

    def compute_grating(self, xrot,
                        spatial_freq=0.1, contrast=1, time_phase=0.):
        return contrast*(1+np.cos(2*np.pi*(spatial_freq*xrot-time_phase)))/2.

    ################################
    #  ---     Experiment      --- #
    ################################

    def init_experiment(self, protocol, keys, run_type='static'):

        self.experiment, self.PATTERNS = {}, []

        if protocol['Presentation']=='Single-Stimulus':
            # single stimulus type
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

        else:
            # ------------  MULTIPLE STIMS ------------
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

            for k in ['index', 'repeat','time_start', 'time_stop',
                    'interstim', 'time_duration', 'interstim-screen', 'frame_run_type']:
                self.experiment[k] = []

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

        for k in ['index', 'repeat','time_start', 'time_stop',
                    'interstim', 'time_duration', 'interstim-screen', 'frame_run_type']:
            self.experiment[k] = np.array(self.experiment[k]) 

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
    # adding a run purely define by an array (time, x, y), see e.g. sparse_noise initialization
    def array_sequence_presentation(self, parent, index):
        tic = time.time()
        # print('stim_index', self.experiment['index'][index])
        # -------------------------------------------------------
        time_indices, frames, refresh_freq = self.get_frames_sequence(index) # refresh_freq can be stimulus dependent !
        print('  array init took %.1fs' % (time.time()-tic))
        toc = time.time()
        FRAMES = []
        for frame in frames:
            FRAMES.append(visual.ImageStim(self.win,
                                           image=self.gamma_corrected_lum(frame),
                                           units='pix', size=self.win.size))
        print('  array buffering took %.1fs' % (time.time()-toc))
        print('  full episode init took %.1fs' % (time.time()-tic))
        self.buffer_delay = np.max([self.buffer_delay, time.time()-tic])
        start = clock.getTime()
        while ((clock.getTime()-start)<(self.experiment['time_duration'][index])) and not parent.stop_flag:
            iframe = int((clock.getTime()-start)*refresh_freq) # refresh_freq can be stimulus dependent !
            if iframe>=len(time_indices):
                print('for index:')
                print(iframe, len(time_indices), len(frames), refresh_freq)
                print(' /!\ Pb with time indices index  /!\ ')
                print('forcing lower values')
                iframe = len(time_indices)-1
            FRAMES[time_indices[iframe]].draw()
            self.add_monitoring_signal(clock.getTime(), start)
            try:
                self.win.flip()
            except AttributeError:
                pass


    #####################################################
    # adding a run purely define by an array -- NOW BUFFERED 
    def buffer_stim(self, parent, gui_refresh_func=None):
        """
        we build the buffers order so that we can call them as:
        self.buffer[protocol_index][stim_index] 
        where:
        protocol_index = stim.experiment['protocol_id'][index]
        stim_index = stim.experiment['index'][index]
        where "index" is the episode number over the full protocol run (including multiprotocols)
        """
        cls = (parent if parent is not None else self)
        win = cls.win if hasattr(cls, 'win') else self.win

        self.buffer = []
        if 'protocol_id' in self.experiment:
            protocol_ids = self.experiment['protocol_id']
        else:
            protocol_ids = np.zeros(len(self.experiment['index']), dtype=int)

        print(' --> buffering stimuli [...] ') 
        tic = time.time()
        for protocol_id in np.sort(np.unique(protocol_ids)):
            self.buffer.append([]) # adding a new set of buffers
            print('    - protocol %i  ' % (protocol_id+1)) 
            single_indices = np.arange(len(protocol_ids))[(protocol_ids==protocol_id) & (self.experiment['repeat']==0)] # this gives the valid
            indices_order = np.argsort(self.experiment['index'][single_indices])
            for stim_index, index_in_full_array in enumerate(single_indices[indices_order]):
                toc = time.time()
                time_indices, frames, refresh_freq = self.get_frames_sequence(index_in_full_array)
                self.buffer[protocol_id].append({'time_indices':time_indices,
                                                 'frames':frames,
                                                 'FRAMES':[],
                                                 'refresh_freq':refresh_freq})
                for frame in self.buffer[protocol_id][stim_index]['frames']:
                    self.buffer[protocol_id][stim_index]['FRAMES'].append(visual.ImageStim(win,
                                                                          image=self.gamma_corrected_lum(frame),
                                                                          units='pix', size=win.size))
                    if gui_refresh_func is not None:    
                        gui_refresh_func()
                print('        index #%i   (%.2fs)' % (stim_index+1, time.time()-toc)) 
   
        print(' --> buffering done ! (t=%.2fs / %.2fmin)' % (time.time()-tic, (time.time()-tic)/60.)) 
        return True

    def array_sequence_buffered_presentation(self, parent, index):
        # --- fetch protocol_id and stim_index:
        protocol_id = self.experiment['protocol_id'][index] if 'protocol_id' in self.experiment else 0
        stim_index = self.experiment['index'][index]
        # print('stim_index', stim_index)
        # -------------------------------------------------------
        # then run loop over buffered frames
        start = clock.getTime()
        while ((clock.getTime()-start)<(self.experiment['time_duration'][index])) and not parent.stop_flag:
            iframe = int((clock.getTime()-start)*self.buffer[protocol_id][stim_index]['refresh_freq'])
            #print(self.buffer[protocol_id][stim_index]['refresh_freq'])
            #print(iframe, len(self.buffer[protocol_id][stim_index]['time_indices']))
            self.buffer[protocol_id][stim_index]['FRAMES'][self.buffer[protocol_id][stim_index]['time_indices'][iframe]].draw()
            self.add_monitoring_signal(clock.getTime(), start)
            try:
                self.win.flip()
            except AttributeError:
                pass


    ## FINAL RUN FUNCTION
    def run(self, parent=None):

        try:
            t0 = np.load(os.path.join(str(parent.datafolder.get()), 'NIdaq.start.npy'))[0]
        except FileNotFoundError:
            print(str(parent.datafolder.get()), 'NIdaq.start.npy', 'not found !')
            t0 = time.time()

        self.start_screen(parent) 

        if ('buffer' in self.protocol) and self.protocol['buffer'] and (self.buffer is None):
            self.buffer_stim(parent)

        for i in range(len(self.experiment['index'])):
            if stop_signal(parent):
                break
            t = time.time()-t0
            print('t=%.2dh:%.2dm:%.2fs - Running protocol of index %i/%i                                protocol-ID:%i' % (t/3600,
                (t%3600)/60, (t%60), i+1, len(self.experiment['index']),
                 self.experiment['protocol_id'][i] if 'protocol_id' in self.experiment else 0))

            # ---- single_episode_run ----- #
            if self.buffer is not None:
                self.array_sequence_buffered_presentation(parent, i) # buffered version
            else: # non-buffered by defaults
                self.array_sequence_presentation(parent, i) # non-buffered version

            if i<(len(self.experiment['index'])-1):
                self.inter_screen(parent,
                                  duration=1.*self.experiment['interstim'][i],
                                  color=self.experiment['interstim-screen'][i])
        self.end_screen(parent)
        if not parent.stop_flag and hasattr(parent, 'statusBar'):
            parent.statusBar.showMessage('stimulation over !')


    ##########################################################
    #############    DRAWING STIMULI (offline)  ##############
    ##########################################################

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        print('to be implemented in child class')
        return 0*self.x

    def plot_stim_picture(self, episode,
                          ax=None, parent=None,
                          label=None, vse=False):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode,
                             ax=ax,
                             label=label,
                             vse=vse,
                             parent=parent)

        return ax

    def get_prestim_image(self):
        return (1+self.protocol['presentation-prestim-screen'])/2.+0*self.x
    def get_interstim_image(self):
        return (1+self.protocol['presentation-interstim-screen'])/2.+0*self.x
    def get_poststim_image(self):
        return (1+self.protocol['presentation-poststim-screen'])/2.+0*self.x

    def image_to_frame(self, img, norm=False, psychopy_to_numpy=False):
        """ need to transpose given the current coordinate system"""
        if psychopy_to_numpy:
            return img.T/2.+0.5
        if norm:
            return (img.T-np.min(img))/(np.max(img)-np.min(img)+1e-6)
        else:
            return img.T

    def get_vse(self, episode, parent=None):
        """
        Virtual Scene Exploration dictionary
        None by default, should be overriden by method in children class
        """
        return None

    def show_frame(self, episode,
                   time_from_episode_start=0,
                   parent=None,
                   label={'degree':5,
                          'shift_factor':0.02,
                          'lw':2, 'fontsize':12},
                   arrow=None,
                   vse=False,
                   ax=None,
                   return_img=False):
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

        cls = (parent if parent is not None else self)
        
        try:
            img = ax.imshow(cls.image_to_frame(cls.get_image(episode,
                                                       time_from_episode_start=time_from_episode_start,
                                                       parent=cls), psychopy_to_numpy=True),
                            extent=(0, self.screen['resolution'][0], 0, self.screen['resolution'][1]),
                      cmap='gray', vmin=0, vmax=1,
                      origin='lower',
                      aspect='equal')
        except BaseException as be:
            print(be)
            print(' /!\ pb in get stim /!\  ')

        if vse:
            self.vse = self.get_vse(episode, parent=cls)
            if self.vse is not None:
                self.add_vse(ax, self.vse)

        ax.axis('off')

        if label is not None:
            nz, nx = self.x.shape
            L, shift = nx/(self.x[-1][-1]-self.x[0][0])*label['degree'], label['shift_factor']*nx
            ax.plot([-shift, -shift], [-shift,L-shift], 'k-', lw=label['lw'])
            ax.plot([-shift, L-shift], [-shift,-shift], 'k-', lw=label['lw'])
            ax.annotate('%.0f$^o$ ' % label['degree'], (-shift, -shift), fontsize=label['fontsize'], ha='right', va='bottom')

        if return_img:
            return img
        else:
            return ax

    def update_frame(self, episode, img,
                     time_from_episode_start=0,
                     parent=None):
        cls = (parent if parent is not None else self)
        
        img.set_array(cls.image_to_frame(cls.get_image(episode,
                                                      time_from_episode_start=time_from_episode_start,
                                                     parent=cls), psychopy_to_numpy=True))


    def add_arrow(self, arrow, ax):
        nz, nx = self.x.shape
        ax.arrow(self.angle_to_pix(arrow['center'][0])+nx/2,
                 self.angle_to_pix(arrow['center'][1])+nz/2,
                 np.cos(np.pi/180.*arrow['direction'])*self.angle_to_pix(arrow['length']),
                 -np.sin(np.pi/180.*arrow['direction'])*self.angle_to_pix(arrow['length']),
                 width=self.angle_to_pix(arrow['length'])*arrow['width_factor'],
                 color=arrow['color'])

    def add_vse(self, ax, vse):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.plot([self.screen['resolution'][0]/2.]+list(vse['x']),
                [self.screen['resolution'][1]/2.]+list(vse['y']), 'o-', color='#d62728', lw=0.5, ms=2)


#####################################################
##  ----      MOVIE STIMULATION REPLAY      --- #####
#####################################################

class movie_replay(visual_stim):
    """ TO BE IMPLEMENTED """

    def __init__(self, protocol):

        super().__init__(protocol)

    def run(self, parent):
        pass



#####################################################
##  ----         MULTI-PROTOCOLS            --- #####
#####################################################

class multiprotocol(visual_stim):

    def __init__(self, protocol):

        super().__init__(protocol, 
                         demo=(('demo' in protocol) and protocol['demo']))

        self.STIM, i = [], 1

        if ('load_from_protocol_data' in protocol) and protocol['load_from_protocol_data']:
            while 'Protocol-%i'%i in protocol:
                subprotocol = {'Screen':protocol['Screen'],
                               'Presentation':'',
                               'demo':(('demo' in protocol) and protocol['demo']),
                               'no-window':True}
                for key in protocol:
                    if ('Protocol-%i-'%i in key):
                        subprotocol[key.replace('Protocol-%i-'%i, '')] = protocol[key]
                self.STIM.append(build_stim(subprotocol))
                i+=1
        else:
            while 'Protocol-%i'%i in protocol:
                path_list = [pathlib.Path(__file__).resolve().parents[1], 'exp', 'protocols']+protocol['Protocol-%i'%i].split('/')
                Ppath = os.path.join(*path_list)
                if not os.path.isfile(Ppath):
                    print(' /!\ "%s" not found in Protocol folder /!\  ' % protocol['Protocol-%i'%i])
                with open(Ppath, 'r') as fp:
                    subprotocol = json.load(fp)
                    subprotocol['Screen'] = protocol['Screen']
                    subprotocol['no-window'] = True
                    subprotocol['demo'] = (('demo' in protocol) and protocol['demo'])
                    self.STIM.append(build_stim(subprotocol))
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

        # ---------------------------- #
        # # SHUFFLING IF NECESSARY
        # ---------------------------- #

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

        for key in ['protocol_id', 'index', 'repeat', 'interstim', 'time_start', 'time_stop', 'time_duration']:
            self.experiment[key] = np.array(self.experiment[key])

    # functions implemented in child class
    def get_frame(self, index):
        return self.STIM[self.experiment['protocol_id'][index]].get_frame(index, parent=self)
    def get_patterns(self, index):
        return self.STIM[self.experiment['protocol_id'][index]].get_patterns(index, parent=self)
    def get_frames_sequence(self, index):
        return self.STIM[self.experiment['protocol_id'][index]].get_frames_sequence(index, parent=self)
    def get_image(self, episode, time_from_episode_start=0, parent=None):
        return self.STIM[self.experiment['protocol_id'][episode]].get_image(episode, time_from_episode_start=time_from_episode_start, parent=self)
    def plot_stim_picture(self, episode, ax=None, parent=None, label=None, vse=False):
        return self.STIM[self.experiment['protocol_id'][episode]].plot_stim_picture(episode, ax=ax, parent=self, label=label, vse=vse)
    def get_vse(self, episode, ax=None, parent=None, label=None, vse=False):
        return self.STIM[self.experiment['protocol_id'][episode]].get_vse(episode, parent=self)



#####################################
##  ----  BUILDING STIMULI  --- #####
#####################################

def init_bg_image(cls, index):
    """ initializing an empty image"""
    return 2*cls.experiment['bg-color'][index]-1.+0.*cls.x

def init_times_frames(cls, index, refresh_freq, security_factor=1.5):
    """ we use this function for each protocol initialisation"""
    interval = cls.experiment['time_stop'][index]-cls.experiment['time_start'][index]
    itend = int(security_factor*interval*refresh_freq)
    return np.arange(itend), np.arange(itend)/refresh_freq, []


class vis_stim_image_built(visual_stim):

    """
    in this object we do not use the psychopy pre-built functions
    to present stimuli
    we rather build the image manually (with numpy) and we show a sequence of ImageStim
    """

    def __init__(self, protocol,
		 keys=['bg-color', 'contrast']):

        super().__init__(protocol)

        if ('buffer' in self.protocol) and (self.protocol['buffer']=="True"):
            super().init_experiment(protocol, keys,
                                    run_type='images_sequence_buffered')
        else:
            super().init_experiment(protocol, keys,
                                    run_type='images_sequence')


        # dealing with refresh rate
        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 10.

        self.refresh_freq = protocol['movie_refresh_freq']
        # adding a appearance threshold (see blob stim)
        if 'appearance_threshold' not in protocol:
            protocol['appearance_threshold'] = 2.5 # 

    def get_image(self):
        print('should be implemented in child class ! ')
        return 0.5+0*self.x # grey screen by default


    def get_frames_sequence(self, index, parent=None):
        """
        """
        cls = (parent if parent is not None else self)
        time_indices, times, FRAMES = init_times_frames(cls, index, self.refresh_freq)
        for iframe, t in enumerate(times):
            FRAMES.append(self.image_to_frame(self.get_image(index,
                                                             time_from_episode_start=t,
                                                             parent=parent)))
        return time_indices, FRAMES, self.refresh_freq


    def compute_frame_order(self, cls, times, index):
        """
        """   
        order = np.arange(len(times))
        if ('randomize' in self.protocol) and (self.protocol['randomize']=="True"):
            # we randomize the order of the time sequence here !!
            if ('randomize-per-trial' in self.protocol) and (self.protocol['randomize-per-trial']=="True"):
                np.random.seed(int(cls.experiment['seed'][index]+1000*index))
            else:
                np.random.seed(int(cls.experiment['seed'][index]))
            np.random.shuffle(order) # shuffling
        return order


    def add_grating_patch(self, image,
                          angle=0,
                          radius=10,
                          spatial_freq=0.1,
                          contrast=1.,
                          time_phase=0.,
                          xcenter=0,
                          zcenter=0):
        """ add a grating patch, drifting when varying the time phase"""
        xrot = self.compute_rotated_coords(angle,
                                           xcenter=xcenter,
                                           zcenter=zcenter)

        cond = ((self.x-xcenter)**2+(self.z-zcenter)**2)<radius**2

        #image[cond] = 2*self.compute_grating(xrot[cond],
        #                                     spatial_freq=spatial_freq,
        #                                     contrast=contrast,
        #                                     time_phase=time_phase)-1

        full_grating = self.compute_grating(xrot,
                                            spatial_freq=spatial_freq,
                                            contrast=1,
                                            time_phase=time_phase)-0.5

        # image[cond] += 2*contrast*full_grating[cond] # /!\ "+=" to see both 
        image[cond] = 2*contrast*full_grating[cond] # /!\ "=" for the patch 



    def add_gaussian(self, image,
                     t=0, t0=0, sT=1.,
                     radius=10,
                     contrast=1.,
                     xcenter=0,
                     zcenter=0):
        """ add a gaussian luminosity increase
        N.B. when contrast=1, you need black background, otherwise it will saturate
             when contrast=0.5, you can start from the grey background to reach white in the center
        """
        image += 2*np.exp(-((self.x-xcenter)**2+(self.z-zcenter)**2)/2./radius**2)*\
                     contrast*np.exp(-(t-t0)**2/2./sT**2)


    def add_dot(self, image, pos, size, color, type='square'):
        """
        add dot, either square or circle
        """
        if type=='square':
            cond = (self.x>(pos[0]-size/2)) & (self.x<(pos[0]+size/2)) & (self.z>(pos[1]-size/2)) & (self.z<(pos[1]+size/2))
        else:
            cond = np.sqrt((self.x-pos[0])**2+(self.z-pos[1])**2)<size
        image[cond] = color



    def new(self):
        pass


#####################################################
##  ----      PRESENTING UNIFORM BACKGROUND --- #####
#####################################################


class uniform_bg(vis_stim_image_built):
    """
    """
    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['bg-color'])

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        return init_bg_image(cls, episode)

    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, vse=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode, ax=ax, label=label,
                             parent=parent)
        return ax


#####################################################
##  ----      PRESENTING GRATING STIMS      --- #####
#####################################################


class center_grating(vis_stim_image_built):
    """
    """
    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['bg-color',
                               'x-center', 'y-center',
                               'radius','spatial-freq',
                               'angle', 'contrast'])


    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        img = init_bg_image(cls, episode)
        self.add_grating_patch(img,
                       angle=cls.experiment['angle'][episode],
                       radius=cls.experiment['radius'][episode],
                       spatial_freq=cls.experiment['spatial-freq'][episode],
                       contrast=cls.experiment['contrast'][episode],
                       xcenter=cls.experiment['x-center'][episode],
                       zcenter=cls.experiment['y-center'][episode])
        return img

    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, vse=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode, ax=ax, label=label,
                             parent=parent)
        return ax


class center_drifting_grating(vis_stim_image_built):
    """
    """
    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['bg-color', 'speed',
                               'x-center', 'y-center',
                               'radius','spatial-freq',
                               'angle', 'contrast'])

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        img = init_bg_image(cls, episode)
        self.add_grating_patch(img,
                       angle=cls.experiment['angle'][episode],
                       radius=cls.experiment['radius'][episode],
                       spatial_freq=cls.experiment['spatial-freq'][episode],
                       contrast=cls.experiment['contrast'][episode],
                       xcenter=cls.experiment['x-center'][episode],
                       zcenter=cls.experiment['y-center'][episode],
                       time_phase=cls.experiment['speed'][episode]*time_from_episode_start)
        return img

    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, vse=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode, ax=ax, label=label,
                             parent=parent)
        arrow['direction'] = cls.experiment['direction'][episode]
        arrow['center'] = [cls.experiment['x-center'][episode],
                           cls.experiment['y-center'][episode]]
        self.add_arrow(arrow, ax)
        return ax
drifting_center_grating = center_drifting_grating


#####################################################
##  ----    PRESENTING MOVING DOTS          --- #####
#####################################################

def get_starting_point_and_direction_mv_dots(cls, index):

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


class line_moving_dots(vis_stim_image_built):
    """
    """
    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['speed', 'bg-color', 'ndots', 'spacing',
                               'direction', 'size', 'dotcolor', 'seed'])

        ## /!\ here always use self.refresh_freq not the parent cls.refresh_freq ##
        # when the parent multiprotocol will have ~10Hz refresh rate, the random case should remain 2-3Hz
        self.refresh_freq = protocol['movie_refresh_freq']


    def get_frames_sequence(self, index, parent=None):
        """
        We overwrite the method (easier than finding an equation
        to have the frame at time t)
        """
        cls = (parent if parent is not None else self)

        X0, Y0, dx_per_time, dy_per_time = get_starting_point_and_direction_mv_dots(cls, index)

        time_indices, times, FRAMES = init_times_frames(cls, index, self.refresh_freq)

        order = self.compute_frame_order(cls, times, index) # shuffling inside if randomize !!

        for iframe, t in enumerate(times):
            new_t = order[iframe]/self.refresh_freq
            img = init_bg_image(cls, index)
            for x0, y0 in zip(X0, Y0):
                # adding the dots one by one
                new_position = (x0+dx_per_time*new_t, y0+dy_per_time*new_t)
                self.add_dot(img, new_position,
                             cls.experiment['size'][index],
                             cls.experiment['dotcolor'][index])

            FRAMES.append(self.image_to_frame(img))

        return time_indices, FRAMES, self.refresh_freq


    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        img = init_bg_image(cls, episode)
        X0, Y0, dx_per_time, dy_per_time = get_starting_point_and_direction_mv_dots(cls, episode)
        for x0, y0 in zip(X0, Y0):
            new_position = (X0+dx_per_time*time_from_episode_start,
                            Y0+dy_per_time*time_from_episode_start)
            self.add_dot(img, new_position,
                         cls.experiment['size'][episode],
                         cls.experiment['dotcolor'][episode])
        return img


    def plot_stim_picture(self, episode,
                          ax=None, parent=None, 
                          label=None, vse=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        cls = (parent if parent is not None else self)
        direction = cls.experiment['direction'][episode]

        png_name = 'line-mv-dots_direction-%i.png' %  direction
        filename = os.path.join(str(pathlib.Path(__file__).resolve().parents[2]),
                                'doc', 'stimuli', png_name)

        if os.path.isfile(filename):
            ge.image(filename, ax=ax)

        # cls = (parent if parent is not None else self)
        # tcenter_minus = .2*(cls.experiment['time_stop'][episode]-\
                             # cls.experiment['time_start'][episode])
        # ax = self.show_frame(episode, ax=ax, label=label,
                             # time_from_episode_start=tcenter_minus,
                             # parent=parent)

        # direction = cls.experiment['direction'][episode]
        # if direction==0:
            # print(0)

        # # print(direction)
        # arrow['direction'] = ((direction+180)%180)+180
        # # print(arrow['direction'])

        # for shift in [-.5, 0, .5]:
            # arrow['center'] = [shift*np.sin(np.pi/180.*direction)*np.max(cls.x)/3.,
                               # shift*np.cos(np.pi/180.*direction)*np.max(cls.z)/3.]
            # self.add_arrow(arrow, ax)

        return ax

class mixed_moving_dots_static_patch(vis_stim_image_built):

    def __init__(self, protocol):

        super().__init__(protocol,
                         ['speed', 'bg-color', 'ndots', 'spacing',
                          'direction', 'size', 'dotcolor', 'seed',
                          'patch-delay', 'patch-duration',
                          'patch-radius', 'patch-contrast',
                          'patch-spatial-freq', 'patch-angle'])


    def get_frames_sequence(self, index, parent=None):
        """
        get frame seq
        """
        cls = (parent if parent is not None else self)

        X0, Y0, dx_per_time, dy_per_time = get_starting_point_and_direction_mv_dots(cls, index)

        time_indices, times, FRAMES = init_times_frames(cls, index, self.refresh_freq)

        order = self.compute_frame_order(cls, times, index) # shuffling inside if randomize !!

        for iframe, t in enumerate(times):
            new_t = order[iframe]/self.refresh_freq
            img = init_bg_image(cls, index)
            for x0, y0 in zip(X0, Y0):
                # adding the dots one by one
                new_position = (x0+dx_per_time*new_t, y0+dy_per_time*new_t)
                self.add_dot(img, new_position,
                             cls.experiment['size'][index],
                             cls.experiment['dotcolor'][index])
            if (t>=(cls.experiment['patch-delay'][index])) and\
                        (t<=(cls.experiment['patch-delay'][index]+cls.experiment['patch-duration'][index])):
                 self.add_grating_patch(img,
                                angle=cls.experiment['patch-angle'][index],
                                radius=cls.experiment['patch-radius'][index],
                                spatial_freq=cls.experiment['patch-spatial-freq'][index],
                                contrast=cls.experiment['patch-contrast'][index])

            FRAMES.append(self.image_to_frame(img))

        return time_indices, FRAMES, self.refresh_freq


    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        img = init_bg_image(cls, episode)
        X0, Y0, dx_per_time, dy_per_time = get_starting_point_and_direction_mv_dots(cls, episode)
        for x0, y0 in zip(X0, Y0):
            new_position = (x0+dx_per_time*time_from_episode_start,
                            y0+dy_per_time*time_from_episode_start)
            self.add_dot(img, new_position,
                         cls.experiment['size'][episode],
                         cls.experiment['dotcolor'][episode])
        self.add_grating_patch(img,
                       angle=cls.experiment['patch-angle'][episode],
                       radius=cls.experiment['patch-radius'][episode],
                       spatial_freq=cls.experiment['patch-spatial-freq'][episode],
                       contrast=cls.experiment['patch-contrast'][episode])
        return img

    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, 
                          vse=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        cls = (parent if parent is not None else self)
        tcenter_minus = .3*(cls.experiment['time_stop'][episode]-\
                             cls.experiment['time_start'][episode])
        ax = self.show_frame(episode, ax=ax, label=label, 
                             time_from_episode_start=tcenter_minus,
                             parent=parent)

        direction = cls.experiment['direction'][episode]

        arrow['direction'] = direction
        # arrow['direction'] = ((direction+180)%180)+180
        # print(arrow['direction'])

        for shift in [-1, 0, 1]:
            arrow['center'] = [shift*np.sin(np.pi/180.*direction)*np.max(cls.x)/3.,
                               shift*np.cos(np.pi/180.*direction)*np.max(cls.z)/3.]
            self.add_arrow(arrow, ax)

        return ax

#####################################################
##  ----    PRESENTING RANDOM DOTS          --- #####
#####################################################

def compute_new_image_with_dots(cls, index,
                                away_from_edges_factor=4):

    dot_size_pix = int(np.round(cls.angle_to_pix(cls.experiment['size'][index]),0))
    Nx = int(cls.x.shape[0]/dot_size_pix)
    Nz = int(cls.x.shape[1]/dot_size_pix)

    img = init_bg_image(cls, index)
    for n in range(cls.experiment['ndots'][index]):
        ix, iz = (np.random.choice(np.arange(away_from_edges_factor, Nx-away_from_edges_factor)[::2],1, replace=False)[0],
                np.random.choice(np.arange(away_from_edges_factor,Nz-away_from_edges_factor)[::2],1, replace=False)[0])
        img[dot_size_pix*ix:dot_size_pix*(ix+1),
            dot_size_pix*iz:dot_size_pix*(iz+1)] = cls.experiment['dotcolor'][index]
    return img
 


class random_dots(vis_stim_image_built):
    """
    simple random dots 
    """
    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['bg-color', 'ndots', 'size', 'dotcolor', 'seed'])

        ## /!\ here always use self.refresh_freq not the parent cls.refresh_freq ##
        # when the parent multiprotocol will have ~10Hz refresh rate, the random case should remain 2-3Hz
        self.refresh_freq = protocol['movie_refresh_freq']

    def get_frames_sequence(self, index, parent=None):
        """
        get frame seq
        """

        cls = (parent if parent is not None else self)

        time_indices, times, FRAMES = init_times_frames(cls, index, self.refresh_freq)

        np.random.seed(int(cls.experiment['seed'][index]+3*index)) # changing seed at each realization
        for iframe, t in enumerate(times):
            img = compute_new_image_with_dots(cls, index)
            FRAMES.append(self.image_to_frame(img))

        return time_indices, FRAMES, self.refresh_freq


    def get_image(self, episode, time_from_episode_start=0, parent=None):
        return self.compute_new_image_with_dots(index)


    def plot_stim_picture(self, episode,
                          ax=None, parent=None, 
                          label=None, vse=False,
                          arrow=None):

        cls = (parent if parent is not None else self)

        ax = self.show_frame(episode, ax=ax, label=label,
                             parent=parent)

        return ax


class mixed_random_dots_static_patch(vis_stim_image_built):

    def __init__(self, protocol):

        super().__init__(protocol,
                         ['bg-color', 'ndots',
                          'size', 'dotcolor', 'seed',
                          'patch-delay', 'patch-duration',
                          'patch-radius', 'patch-contrast',
                          'patch-spatial-freq', 'patch-angle'])


    def get_frames_sequence(self, index, parent=None):
        """
        get frame seq
        """
        cls = (parent if parent is not None else self)

        time_indices, times, FRAMES = init_times_frames(cls, index, self.refresh_freq)

        np.random.seed(int(cls.experiment['seed'][index]+3*index)) # changing seed at each realization
        for iframe, t in enumerate(times):
            # random dot frame
            img = compute_new_image_with_dots(cls, index)
            # on which we add the patch
            if (t>=(cls.experiment['patch-delay'][index])) and\
                        (t<=(cls.experiment['patch-delay'][index]+cls.experiment['patch-duration'][index])):
                 self.add_grating_patch(img,
                                angle=cls.experiment['patch-angle'][index],
                                radius=cls.experiment['patch-radius'][index],
                                spatial_freq=cls.experiment['patch-spatial-freq'][index],
                                contrast=cls.experiment['patch-contrast'][index])

            FRAMES.append(self.image_to_frame(img))

        return time_indices, FRAMES, self.refresh_freq


    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        img = init_bg_image(cls, episode)
        X0, Y0, dx_per_time, dy_per_time = get_starting_point_and_direction_mv_dots(cls, episode)
        for x0, y0 in zip(X0, Y0):
            new_position = (x0+dx_per_time*time_from_episode_start,
                            y0+dy_per_time*time_from_episode_start)
            self.add_dot(img, new_position,
                         cls.experiment['size'][episode],
                         cls.experiment['dotcolor'][episode])
        self.add_grating_patch(img,
                       angle=cls.experiment['patch-angle'][episode],
                       radius=cls.experiment['patch-radius'][episode],
                       spatial_freq=cls.experiment['patch-spatial-freq'][episode],
                       contrast=cls.experiment['patch-contrast'][episode])
        return img

    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, 
                          vse=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        cls = (parent if parent is not None else self)
        tcenter_minus = .3*(cls.experiment['time_stop'][episode]-\
                             cls.experiment['time_start'][episode])
        ax = self.show_frame(episode, ax=ax, label=label, 
                             time_from_episode_start=tcenter_minus,
                             parent=parent)

        direction = cls.experiment['direction'][episode]

        arrow['direction'] = direction
        # arrow['direction'] = ((direction+180)%180)+180
        # print(arrow['direction'])

        for shift in [-1, 0, 1]:
            arrow['center'] = [shift*np.sin(np.pi/180.*direction)*np.max(cls.x)/3.,
                               shift*np.cos(np.pi/180.*direction)*np.max(cls.z)/3.]
            self.add_arrow(arrow, ax)

        return ax



#####################################################
##  ----    PRESENTING NATURAL IMAGES       --- #####
#####################################################


def get_NaturalImages_as_array(screen):
    
    NI_FOLDERS = [os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'NI_bank'),
                  os.path.join(os.path.expanduser('~'), 'physion', 'physion', 'visual_stim', 'NI_bank'),
                  os.path.join(os.path.expanduser('~'), 'work', 'physion', 'physion', 'visual_stim', 'NI_bank')]
    
    NIarray = []

    NI_directory = None
    for d in NI_FOLDERS:
        if os.path.isdir(d):
            NI_directory = d

    if NI_directory is not None:
        for filename in np.sort(os.listdir(NI_directory)):
            img = load(os.path.join(NI_directory, filename))
            new_img = adapt_to_screen_resolution(img, screen)
            NIarray.append(2*img_after_hist_normalization(new_img)-1.)
        return NIarray
    else:
        print(' /!\  Natural Images folder not found !!! /!\  ')
        return [np.ones((10,10))*0.5 for i in range(5)]
    

class natural_image(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['Image-ID'], run_type='image_sequence')
        
        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 2 
        self.refresh_freq = protocol['movie_refresh_freq']

        # initializing set of NI
        self.NIarray = get_NaturalImages_as_array(self.screen)

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        return (1.+self.NIarray[int(cls.experiment['Image-ID'][episode])])/2.
            
    def get_frames_sequence(self, index, parent=None):
        cls = (parent if parent is not None else self)
        
        img = self.NIarray[int(cls.experiment['Image-ID'][index])]
            
        interval = cls.experiment['time_stop'][index]-cls.experiment['time_start'][index]
        time_indices, FRAMES = np.zeros(int(2*interval*self.refresh_freq), dtype=int), []
        Times = np.arange(len(time_indices))/self.refresh_freq

        FRAMES.append(img)
        time_indices[Times>=0] = 0
            
        return time_indices, FRAMES, self.refresh_freq

    def plot_stim_picture(self, episode, parent=None, 
                          vse=True, ax=None, label=None,
                          time_from_episode_start=0):

        cls = (parent if parent is not None else self)

        if ax==None:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots(1)

        img = ax.imshow(cls.image_to_frame(cls.get_image(episode,
                        time_from_episode_start=time_from_episode_start,
                        parent=cls).T, psychopy_to_numpy=True),
                      cmap='gray', vmin=0, vmax=1,
                      origin='lower',
                      aspect='equal')

        ax.axis('off')

        return ax


#####################################################
##  --    WITH VIRTUAL SCENE EXPLORATION    --- #####
#####################################################


def generate_VSE(duration=5,
                 min_saccade_duration=1.,# in s
                 max_saccade_duration=3.,# in s
                 # mean_saccade_duration=2.,# in s
                 # std_saccade_duration=1.,# in s
                 saccade_amplitude=100, # in pixels, TO BE PUT IN DEGREES
                 seed=0,
                 verbose=False):
    """
    A simple 'Virtual-Saccadic-Eye' (VSE) generator
    based on temporal and spatial shifts drown form a uniform random distribution
    """
    
    if verbose:
        print('generating Virtual-Scene-Exploration (with seed "%s") [...]' % seed)
    
    np.random.seed(seed)
    
    tsaccades = np.cumsum(np.random.uniform(min_saccade_duration, max_saccade_duration,
                                            size=int(3*duration/(max_saccade_duration-min_saccade_duration))))

    x = np.random.uniform(saccade_amplitude/5., 2*saccade_amplitude, size=len(tsaccades))
    y = np.random.uniform(saccade_amplitude/5., 2*saccade_amplitude, size=len(tsaccades))
    
    return {'t':np.array(list(tsaccades)),
            'x':np.array(list(x)),
            'y':np.array(list(y)),
            'max_amplitude':saccade_amplitude}

            

class Natural_Image_VSE(visual_stim):

    def __init__(self, protocol):

        super().__init__(protocol)
        super().init_experiment(protocol, ['Image-ID', 'VSE-seed',
                                           'min-saccade-duration', 'max-saccade-duration',
                                           'vary-VSE-with-Image', 'saccade-amplitude'],
                                run_type='images_sequence')

        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 20.
        self.refresh_freq = protocol['movie_refresh_freq']

        # initializing set of NI
        self.NIarray = get_NaturalImages_as_array(self.screen)


    def compute_shifted_image(self, img, ix, iy):
        sx, sy = img.shape
        new_im = np.zeros(img.shape)
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
        vse = self.get_vse(index, parent=cls)
        
        img = self.NIarray[int(cls.experiment['Image-ID'][index])]
            
        interval = cls.experiment['time_stop'][index]-cls.experiment['time_start'][index]
        time_indices, FRAMES = np.zeros(int(2*interval*self.refresh_freq), dtype=int), []
        Times = np.arange(len(time_indices))/self.refresh_freq

        for i, t in enumerate(vse['t']):
            FRAMES.append(self.compute_shifted_image(img, int(vse['x'][i]), int(vse['y'][i])))
            time_indices[Times>=t] = int(i)
            
        return time_indices, FRAMES, self.refresh_freq

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        return (1.+self.NIarray[int(cls.experiment['Image-ID'][episode])])/2.

    def get_vse(self, episode, parent=None):
        """
        translate saccades in degree in pixels here
        """
        cls = (parent if parent is not None else self)
        if 'saccade-amplitude' in cls.experiment:
            seed = self.get_seed(episode, parent=cls)
            return generate_VSE(duration=cls.experiment['time_duration'][episode],
                                min_saccade_duration=cls.experiment['min-saccade-duration'][episode],
                                max_saccade_duration=cls.experiment['max-saccade-duration'][episode],
                                saccade_amplitude=cls.angle_to_pix(cls.experiment['saccade-amplitude'][episode]),
                                seed=seed)
        else:
            return None

    def plot_stim_picture(self, episode, parent=None, 
                          vse=True, ax=None, label=None,
                          time_from_episode_start=0):

        cls = (parent if parent is not None else self)

        if ax==None:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots(1)

        img = ax.imshow(cls.image_to_frame(cls.get_image(episode,
                        time_from_episode_start=time_from_episode_start,
                        parent=cls).T, psychopy_to_numpy=True),
                      cmap='gray', vmin=0, vmax=1,
                      origin='lower',
                      aspect='equal')

        self.vse = self.get_vse(episode, parent=cls)
        self.add_vse(ax, self.vse)

        ax.axis('off')

        return ax
        # if label is not None:
            # nz, nx = self.x.shape
            # L, shift = nx/(self.x[0][-1]-self.x[0][0])*label['degree'], label['shift_factor']*nx
            # ax.plot([-shift, -shift], [-shift,L-shift], 'k-', lw=label['lw'])
            # ax.plot([-shift, L-shift], [-shift,-shift], 'k-', lw=label['lw'])
            # ax.annotate('%.0f$^o$ ' % label['degree'], (-shift, -shift), fontsize=label['fontsize'], ha='right', va='bottom')

        # if return_img:
            # return img
        # else:
            # return ax


            

#####################################################
##  -- PRESENTING APPEARING GAUSSIAN BLOBS  --  #####           
#####################################################

class gaussian_blobs(vis_stim_image_built):
    
    def __init__(self, protocol):

        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 5.
        self.refresh_freq = protocol['movie_refresh_freq']

        super().__init__(protocol,
                         ['x-center', 'y-center', 'radius',
                          'center-time', 'extent-time',
                          'contrast', 'bg-color'])

    def get_image(self, index, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        img = init_bg_image(cls, index)
        self.add_gaussian(img,
                          t=time_from_episode_start, 
                          contrast = cls.experiment['contrast'][index],
                          xcenter=cls.experiment['x-center'][index],
                          zcenter=cls.experiment['y-center'][index],
                          radius = cls.experiment['radius'][index],
                          t0=cls.experiment['center-time'][index],
                          sT=cls.experiment['extent-time'][index])
        return img    

    def plot_stim_picture(self, episode,
                          ax=None, parent=None,
                          label=None, vse=False):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode,
                             time_from_episode_start=cls.experiment['center-time'][episode],
                             ax=ax, parent=parent)

        return ax


#####################################################
##  ----    SOME TOOLS TO DEBUG PROTOCOLS   --- #####
#####################################################

class dummy_datafolder:
    def __init__(self):
        pass
    def get(self):
        return tempfile.gettempdir()

class dummy_parent:
    def __init__(self):
        self.stop_flag = False
        self.datafolder = dummy_datafolder()

if __name__=='__main__':

    import json, argparse, tempfile
    from pathlib import Path

    parser=argparse.ArgumentParser()
    parser.add_argument("protocol", type=str)
    parser.add_argument("-b", "--buffered", help="buffer stim", action="store_true")
    parser.add_argument("-i", "--index", help="stim index", type=int, default=0) 
    parser.add_argument("-p", "--plot", help="plot stim", action="store_true")

    args = parser.parse_args()

    if os.path.isfile(args.protocol) and ('.json' in args.protocol):
        with open(args.protocol, 'r') as fp:
            protocol = json.load(fp)
            protocol['demo'] = True
            if args.buffered:
                protocol['buffer'] = True
            else:
                protocol['buffer'] = False 
            parent = dummy_parent()
            if args.plot:
                sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
                from dataviz.datavyz.datavyz import graph_env
                ge = graph_env('screen')
                fig, ax = ge.figure()
                # 
                protocol['no-window'] = True
                stim = build_stim(protocol)
                stim.plot_stim_picture(args.index, ax=ax)
                ge.show()
            else:
                stim = build_stim(protocol)
                stim.run(parent)
                stim.close()
    else:
        print('need to provide a ".json" protocol file as argument !')
