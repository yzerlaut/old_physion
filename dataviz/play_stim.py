"""
functions to replay the visual stimuli
"""

class visual_stim_replay:

    def __init__(self, nwbfile, protocol):

        if (protocol['Stimulus']=='light-level'):
            self.stim = light_level_single_stim(nwbfile, protocol)

    def get_frame(self, time):
        return self.stim.get_frame(time)
    
    #     elif (protocol['Stimulus']=='full-field-grating'):
    #         return full_field_grating_stim(protocol)
    #     elif (protocol['Stimulus']=='center-grating'):
    #         return center_grating_stim(protocol)
    #     elif (protocol['Stimulus']=='off-center-grating'):
    #         return off_center_grating_stim(protocol)
    #     elif (protocol['Stimulus']=='surround-grating'):
    #         return surround_grating_stim(protocol)
    #     elif (protocol['Stimulus']=='drifting-full-field-grating'):
    #         return drifting_full_field_grating_stim(protocol)
    #     elif (protocol['Stimulus']=='drifting-center-grating'):
    #         return drifting_center_grating_stim(protocol)
    #     elif (protocol['Stimulus']=='drifting-off-center-grating'):
    #         return drifting_off_center_grating_stim(protocol)
    #     elif (protocol['Stimulus']=='drifting-surround-grating'):
    #         return drifting_surround_grating_stim(protocol)
    #     elif (protocol['Stimulus']=='Natural-Image'):
    #         return natural_image(protocol)
    #     elif (protocol['Stimulus']=='Natural-Image+VSE'):
    #         return natural_image_vse(protocol)
    #     elif (protocol['Stimulus']=='sparse-noise'):
    #         if protocol['Presentation']=='Single-Stimulus':
    #             return sparse_noise(protocol)
    #         else:
    #             print('Noise stim have to be done as "Single-Stimulus" !')
    # elif (protocol['Stimulus']=='dense-noise'):
    #     if protocol['Presentation']=='Single-Stimulus':
    #         return dense_noise(protocol)
    #     else:
    #         print('Noise stim have to be done as "Single-Stimulus" !')
    # else:
    #     print('Protocol not recognized !')
    #     return None


#####################################################
##  ----   PRESENTING VARIOUS LIGHT LEVELS  --- #####           
#####################################################

class light_level_single_stim:

    def __init__(self, nwbfile, protocol):

        self.times = [0]
        
    def get_frame(self, index):

        print(self.times[index])
        
        
    # def frame_generator(self):
    #     """
    #     Generator creating a random number of chunks (but at most max_chunks) of length chunk_length containing
    #     random samples of sin([0, 2pi]).
    #     """
    #     x, y = self.pixel_meshgrid()
    #     # prestim
    #     for i in range(int(self.protocol['presentation-prestim-period']*self.protocol['movie_refresh_freq'])):
    #         yield np.ones(x.shape)*(2*self.protocol['presentation-prestim-screen']-1)
    #     for episode, start in enumerate(self.experiment['time_start']):
    #         img = self.experiment['light-level'][episode]+0.*x
    #         img = self.gamma_corrected_lum(img)
    #         for i in range(int(self.protocol['presentation-duration']*self.protocol['movie_refresh_freq'])):
    #             self.add_monitoring_signal(x, y, img, i/self.protocol['movie_refresh_freq'], 0)
    #             yield img
    #         if episode<len(self.experiment['time_start'])-1:
    #             # adding interstim
    #             for i in range(int(self.protocol['presentation-interstim-period']*self.protocol['movie_refresh_freq'])):
    #                 yield np.ones(x.shape)*(2*self.protocol['presentation-interstim-screen']-1)
    #     # poststim
    #     for i in range(int(self.protocol['presentation-poststim-period']*self.protocol['movie_refresh_freq'])):
    #         yield np.ones(x.shape)*(2*self.protocol['presentation-poststim-screen']-1)
    #     return        

            
# #####################################################
# ##  ----   PRESENTING FULL FIELD GRATINGS   --- #####           
# #####################################################

# class full_field_grating_stim(visual_stim):

#     def __init__(self, **args):
        
#         super().__init__(**args)
#         if self.task=='generate':
#             super().init_experiment(self.protocol, ['spatial-freq', 'angle', 'contrast'])
        
#     def frame_generator(self):
#         """
#         Generator creating a random number of chunks (but at most max_chunks) of length chunk_length containing
#         random samples of sin([0, 2pi]).
#         """
#         xp, zp = self.pixel_meshgrid()
#         x, z = self.horizontal_pix_to_angle(xp), self.vertical_pix_to_angle(zp)
#         # prestim
#         for i in range(int(self.protocol['presentation-prestim-period']*self.protocol['movie_refresh_freq'])):
#             yield np.ones(x.shape)*(2*self.protocol['presentation-prestim-screen']-1)
#         for episode, start in enumerate(self.experiment['time_start']):
#             angle = self.experiment['angle'][episode]
#             spatial_freq = self.experiment['spatial-freq'][episode]
#             contrast = self.experiment['contrast'][episode]
#             x_rot = x*np.cos(angle/180.*np.pi)+z*np.sin(angle/180.*np.pi)
#             img = np.cos(2*np.pi*spatial_freq*x_rot)
#             img = self.gamma_corrected_lum(img)
#             for i in range(int(self.protocol['presentation-duration']*self.protocol['movie_refresh_freq'])):
#                 self.add_monitoring_signal(xp, zp, img, i/self.protocol['movie_refresh_freq'], 0)
#                 yield img
#             if episode<len(self.experiment['time_start'])-1:
#                 # adding interstim
#                 for i in range(int(self.protocol['presentation-interstim-period']*self.protocol['movie_refresh_freq'])):
#                     yield np.ones(x.shape)*(2*self.protocol['presentation-interstim-screen']-1)
#         # poststim
#         for i in range(int(self.protocol['presentation-poststim-period']*self.protocol['movie_refresh_freq'])):
#             yield np.ones(x.shape)*(2*self.protocol['presentation-poststim-screen']-1)
#         return        

            
# class drifting_full_field_grating_stim(visual_stim):

#     def __init__(self, **args):
        
#         super().__init__(**args)
#         if self.task=='generate':
#             super().init_experiment(self.protocol, ['spatial-freq', 'angle', 'contrast', 'speed'])
        
#     def frame_generator(self):
#         """
#         Generator creating a random number of chunks (but at most max_chunks) of length chunk_length containing
#         random samples of sin([0, 2pi]).
#         """
#         xp, zp = self.pixel_meshgrid()
#         x = self.horizontal_pix_to_angle(xp)
#         z = self.vertical_pix_to_angle(zp)

#         # prestim
#         for i in range(int(self.protocol['presentation-prestim-period']*self.protocol['movie_refresh_freq'])):
#             yield np.ones(x.shape)*(2*self.protocol['presentation-prestim-screen']-1)
#         for episode, start in enumerate(self.experiment['time_start']):
#             angle = self.experiment['angle'][episode]
#             spatial_freq = self.experiment['spatial-freq'][episode]
#             contrast = self.experiment['contrast'][episode]
#             speed = self.experiment['speed'][episode]
#             x_rot = x*np.cos(angle/180.*np.pi)+z*np.sin(angle/180.*np.pi)
#             for i in range(int(self.protocol['presentation-duration']*self.protocol['movie_refresh_freq'])):
#                 img = np.cos(2*np.pi*spatial_freq*x_rot+2*np.pi*speed*i/self.protocol['movie_refresh_freq'])
#                 img = self.gamma_corrected_lum(img)
#                 self.add_monitoring_signal(xp, zp, img, i/self.protocol['movie_refresh_freq'], 0)
#                 yield img
#             if episode<len(self.experiment['time_start'])-1:
#                 # adding interstim
#                 for i in range(int(self.protocol['presentation-interstim-period']*self.protocol['movie_refresh_freq'])):
#                     yield np.ones(x.shape)*(2*self.protocol['presentation-interstim-screen']-1)
#         # poststim
#         for i in range(int(self.protocol['presentation-poststim-period']*self.protocol['movie_refresh_freq'])):
#             yield np.ones(x.shape)*(2*self.protocol['presentation-poststim-screen']-1)
#         return        
            
# #####################################################
# ##  ----    PRESENTING CENTERED GRATINGS    --- #####           
# #####################################################

# class center_grating_stim(visual_stim):
    
#     def __init__(self, **args):
        
#         super().__init__(**args)
#         if self.task=='generate':
#             super().init_experiment(self.protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color'])
        
#     def frame_generator(self):
#         """
#         Generator creating a random number of chunks (but at most max_chunks) of length chunk_length containing
#         random samples of sin([0, 2pi]).
#         """
#         xp, zp = self.pixel_meshgrid()
#         x, z = self.horizontal_pix_to_angle(xp), self.vertical_pix_to_angle(zp)
#         # prestim
#         for i in range(int(self.protocol['presentation-prestim-period']*self.protocol['movie_refresh_freq'])):
#             yield np.ones(x.shape)*(2*self.protocol['presentation-prestim-screen']-1)
#         for episode, start in enumerate(self.experiment['time_start']):
#             angle = self.experiment['angle'][episode]
#             spatial_freq = self.experiment['spatial-freq'][episode]
#             contrast = self.experiment['contrast'][episode]
#             xcenter, zcenter = self.experiment['x-center'][episode],\
#                 self.experiment['y-center'][episode]
#             radius = self.experiment['radius'][episode]
#             bg_color = self.experiment['bg-color'][episode]
#             x_rot = x*np.cos(angle/180.*np.pi)+z*np.sin(angle/180.*np.pi)
#             circle_cond = ((x-xcenter)**2+(z-zcenter)**2<radius**2)
#             img = (2*bg_color-1)*np.ones(x.shape)
#             img[circle_cond] = np.cos(2*np.pi*spatial_freq*x_rot[circle_cond])
#             img = self.gamma_corrected_lum(img)
#             for i in range(int(self.protocol['presentation-duration']*self.protocol['movie_refresh_freq'])):
#                 self.add_monitoring_signal(xp, zp, img, i/self.protocol['movie_refresh_freq'], 0)
#                 yield img
#             if episode<len(self.experiment['time_start'])-1:
#                 # adding interstim
#                 for i in range(int(self.protocol['presentation-interstim-period']*self.protocol['movie_refresh_freq'])):
#                     yield np.ones(x.shape)*(2*self.protocol['presentation-interstim-screen']-1)
#         # poststim
#         for i in range(int(self.protocol['presentation-poststim-period']*self.protocol['movie_refresh_freq'])):
#             yield np.ones(x.shape)*(2*self.protocol['presentation-poststim-screen']-1)
#         return        

            
# class drifting_center_grating_stim(visual_stim):
    
#     def __init__(self, protocol):
#         super().__init__(protocol)
#         super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'speed', 'bg-color'])

#         print(self.experiment['bg-color'])
#         # then manually building patterns
#         for i in range(len(self.experiment['index'])):
#             self.PATTERNS.append([\
#                                   visual.GratingStim(win=self.win,
#                                                      pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
#                                                      size=self.experiment['radius'][i], mask='circle',
#                                                      sf=self.experiment['spatial-freq'][i],
#                                                      ori=self.experiment['angle'][i],
#                                                      contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i]))])


# #####################################################
# ##  ----    PRESENTING OFF-CENTERED GRATINGS    --- #####           
# #####################################################

# class off_center_grating_stim(visual_stim):
    
#     def __init__(self, protocol):
#         super().__init__(protocol)
#         super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color'])

#         # then manually building patterns
#         for i in range(len(self.experiment['index'])):
#             self.PATTERNS.append([\
#                                   visual.GratingStim(win=self.win,
#                                                      size=1000, pos=[0,0],
#                                                      sf=self.experiment['spatial-freq'][i],
#                                                      ori=self.experiment['angle'][i],
#                                                      contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i])),
#                                   visual.GratingStim(win=self.win,
#                                                      pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
#                                                      size=self.experiment['radius'][i],
#                                                      mask='circle', sf=0,
#                                                      color=self.gamma_corrected_lum(self.experiment['bg-color'][i]))])

            
# class drifting_off_center_grating_stim(visual_stim):
    
#     def __init__(self, protocol):
#         super().__init__(protocol)
#         super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color', 'speed'])

#         # then manually building patterns
#         for i in range(len(self.experiment['index'])):
#             self.PATTERNS.append([\
#                                   # Surround grating
#                                   visual.GratingStim(win=self.win,
#                                                      size=1000, pos=[0,0],
#                                                      sf=self.experiment['spatial-freq'][i],
#                                                      ori=self.experiment['angle'][i],
#                                                      contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i])),
#                                   # + center Mask
#                                   visual.GratingStim(win=self.win,
#                                                      pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
#                                                      size=self.experiment['radius'][i],
#                                                      mask='circle', sf=0, contrast=0,
#                                                      color=self.gamma_corrected_lum(self.experiment['bg-color'][i]))])


# #####################################################
# ##  ----    PRESENTING SURROUND GRATINGS    --- #####           
# #####################################################
        
# class surround_grating_stim(visual_stim):

#     def __init__(self, protocol):
#         super().__init__(protocol)
#         super().init_experiment(protocol, ['x-center', 'y-center', 'radius-start', 'radius-end','spatial-freq', 'angle', 'contrast', 'bg-color'])

#         # then manually building patterns
#         for i in range(len(self.experiment['index'])):
#             self.PATTERNS.append([\
#                                   visual.GratingStim(win=self.win,
#                                                      size=1000, pos=[0,0], sf=0,
#                                                      color=self.gamma_corrected_lum(self.experiment['bg-color'][i])),
#                                   visual.GratingStim(win=self.win,
#                                                      pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
#                                                      size=self.experiment['radius-end'][i],
#                                                      mask='circle', 
#                                                      sf=self.experiment['spatial-freq'][i],
#                                                      ori=self.experiment['angle'][i],
#                                                      contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i])),
#                                   visual.GratingStim(win=self.win,
#                                                      pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
#                                                      size=self.experiment['radius-start'][i],
#                                                      mask='circle', sf=0,
#                                                      color=self.gamma_corrected_lum(self.experiment['bg-color'][i]))])

# class drifting_surround_grating_stim(visual_stim):

#     def __init__(self, protocol):
#         super().__init__(protocol)
#         super().init_experiment(protocol, ['x-center', 'y-center', 'radius-start', 'radius-end','spatial-freq', 'angle', 'contrast', 'bg-color', 'speed'])

#         # then manually building patterns
#         for i in range(len(self.experiment['index'])):
#             self.PATTERNS.append([\
#                                   visual.GratingStim(win=self.win,
#                                                      size=1000, pos=[0,0], sf=0,contrast=0,
#                                                      color=self.gamma_corrected_lum(self.experiment['bg-color'][i])),
#                                   visual.GratingStim(win=self.win,
#                                                      pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
#                                                      size=self.experiment['radius-end'][i],
#                                                      mask='circle', 
#                                                      sf=self.experiment['spatial-freq'][i],
#                                                      ori=self.experiment['angle'][i],
#                                                      contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i])),
#                                   visual.GratingStim(win=self.win,
#                                                      pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
#                                                      size=self.experiment['radius-start'][i],
#                                                      mask='circle', sf=0,contrast=0,
#                                                      color=self.gamma_corrected_lum(self.experiment['bg-color'][i]))])
        

# #####################################################
# ##  ----    PRESENTING NATURAL IMAGES       --- #####
# #####################################################

# NI_directory = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'NI_bank')
        
# class natural_image(visual_stim):

#     def __init__(self, protocol):

#         # from visual_stim.psychopy_code.preprocess_NI import load, img_after_hist_normalization
#         # from .preprocess_NI import load, img_after_hist_normalization
        
#         super().__init__(protocol)
#         super().init_experiment(protocol, ['Image-ID'])

#         for i in range(len(self.experiment['index'])):
#             filename = os.listdir(NI_directory)[int(self.experiment['Image-ID'][i])]
#             img = load(os.path.join(NI_directory, filename))
#             img = 2*self.gamma_corrected_contrast(img_after_hist_normalization(img))-1 # normalization + gamma_correction
#             # rescaled_img = adapt_to_screen_resolution(img, (SCREEN[0], SCREEN[1]))

#             self.PATTERNS.append([visual.ImageStim(self.win, image=img.T,
#                                                    units='pix', size=self.win.size)])


# #####################################################
# ##  --    WITH VIRTUAL SCENE EXPLORATION    --- #####
# #####################################################


# def generate_VSE(duration=5,
#                  mean_saccade_duration=2.,# in s
#                  std_saccade_duration=1.,# in s
#                  # saccade_amplitude=50.,
#                  saccade_amplitude=100, # in pixels, TO BE PUT IN DEGREES
#                  seed=0):
#     """
#     to do: clean up the VSE generator
#     """
#     print('generating Virtual-Scene-Exploration [...]')
    
#     np.random.seed(seed)
    
#     tsaccades = np.cumsum(np.clip(mean_saccade_duration+np.abs(np.random.randn(int(1.5*duration/mean_saccade_duration))*std_saccade_duration),
#                                   mean_saccade_duration/2., 1.5*mean_saccade_duration))

#     x = np.array(np.clip(np.random.randn(len(tsaccades))*saccade_amplitude, 0, saccade_amplitude), dtype=int)
#     y = np.array(np.clip(np.random.randn(len(tsaccades))*saccade_amplitude, 0, saccade_amplitude), dtype=int)
    
#     return {'t':np.array([0]+list(tsaccades)),
#             'x':np.array([0]+list(x)),
#             'y':np.array([0]+list(y)),
#             'max_amplitude':saccade_amplitude}

            

# class natural_image_vse(visual_stim):

#     def __init__(self, protocol):

#         super().__init__(protocol)
#         super().init_experiment(protocol,
#                                 ['Image-ID', 'VSE-seed',
#                                  'mean-saccade-duration', 'std-saccade-duration'])

#         print(self.experiment)
#         self.VSEs = [] # array of Virtual-Scene-Exploration
#         for i in range(len(self.experiment['index'])):

#             vse = generate_VSE(duration=protocol['presentation-duration'],
#                                mean_saccade_duration=self.experiment['mean-saccade-duration'][i],
#                                std_saccade_duration=self.experiment['std-saccade-duration'][i],
#                                saccade_amplitude=100, # in pixels, TO BE PUT IN DEGREES
#                                seed=int(self.experiment['VSE-seed'][i]+self.experiment['Image-ID'][i]))

#             self.VSEs.append(vse)
            
#             filename = os.listdir(NI_directory)[int(self.experiment['Image-ID'][i])]
#             img = load(os.path.join(NI_directory, filename))
#             img = 2*self.gamma_corrected_contrast(img_after_hist_normalization(img))-1 # normalization + gamma_correction
#             # rescaled_img = adapt_to_screen_resolution(img, (SCREEN[0], SCREEN[1]))
#             sx, sy = img.T.shape

#             self.PATTERNS.append([])
            
#             IMAGES = []
#             for i in range(len(vse['t'])):
#                 ix, iy = vse['x'][i], vse['y'][i]
#                 new_im = img.T[ix:sx-vse['max_amplitude']+ix,\
#                                iy:sy-vse['max_amplitude']+iy]
#                 self.PATTERNS[-1].append(visual.ImageStim(self.win,
#                                                           image=new_im,
#                                                           units='pix', size=self.win.size))
            
    
# #####################################################
# ##  -- PRESENTING APPEARING GAUSSIAN BLOBS  --  #####           
# #####################################################

# class gaussian_blobs(visual_stim):
    
#     def __init__(self, **args):
        
#         super().__init__(**args)
#         if self.task=='generate':
#             super().init_experiment(self.protocol, ['x-center', 'y-center',
#                                                     'radius','center-time',
#                                                     'extent-time', 'contrast', 'bg-color'])
        
#     def frame_generator(self):
#         """
#         Generator creating a random number of chunks (but at most max_chunks) of length chunk_length containing
#         random samples of sin([0, 2pi]).
#         """
#         xp, zp = self.pixel_meshgrid()
#         x, z = self.horizontal_pix_to_angle(xp), self.vertical_pix_to_angle(zp)
#         # prestim
#         for i in range(int(self.protocol['presentation-prestim-period']*self.protocol['movie_refresh_freq'])):
#             yield np.ones(x.shape)*(2*self.protocol['presentation-prestim-screen']-1)
#         for episode, start in enumerate(self.experiment['time_start']):
#             t0, sT = self.experiment['center-time'][episode], self.experiment['extent-time'][episode]
#             contrast = self.experiment['contrast'][episode]
#             xcenter, zcenter = self.experiment['x-center'][episode],\
#                 self.experiment['y-center'][episode]
#             radius = self.experiment['radius'][episode]
#             bg_color = self.experiment['bg-color'][episode]
#             for i in range(int(self.protocol['presentation-duration']*self.protocol['movie_refresh_freq'])):
#                 img = 2*(np.exp(-((x-xcenter)**2+(z-zcenter)**2)/2./radius**2)*\
#                     contrast*np.exp(-(i/self.protocol['movie_refresh_freq']-t0)**2/2./sT**2)+bg_color)-1.
#                 img = self.gamma_corrected_lum(img)
#                 self.add_monitoring_signal(xp, zp, img, i/self.protocol['movie_refresh_freq'], 0)
#                 yield img
#             if episode<len(self.experiment['time_start'])-1:
#                 # adding interstim
#                 for i in range(int(self.protocol['presentation-interstim-period']*self.protocol['movie_refresh_freq'])):
#                     yield np.ones(x.shape)*(2*self.protocol['presentation-interstim-screen']-1)
#         # poststim
#         for i in range(int(self.protocol['presentation-poststim-period']*self.protocol['movie_refresh_freq'])):
#             yield np.ones(x.shape)*(2*self.protocol['presentation-poststim-screen']-1)
#         return        

# #####################################################
# ##  ----    PRESENTING BINARY NOISE         --- #####
# #####################################################

# class sparse_noise(visual_stim):
    
#     def __init__(self, protocol):

#         super().__init__(protocol)
#         super().init_experiment(protocol,
#             ['square-size', 'sparseness', 'mean-refresh-time', 'jitter-refresh-time'])
        
#         self.STIM = build_sparse_noise(protocol['presentation-duration'],
#                                        self.monitor,
#                                        square_size=protocol['square-size (deg)'],
#                                        noise_mean_refresh_time=protocol['mean-refresh-time (s)'],
#                                        noise_rdm_jitter_refresh_time=protocol['jitter-refresh-time (s)'],
#                                        seed=protocol['noise-seed (#)'])
        
#         self.experiment = {'refresh-times':self.STIM['t']}
            

# class dense_noise(visual_stim):

#     def __init__(self, protocol):

#         super().__init__(protocol)
#         super().init_experiment(protocol,
#                 ['square-size', 'sparseness', 'mean-refresh-time', 'jitter-refresh-time'])

#         self.STIM = build_dense_noise(protocol['presentation-duration'],
#                                       self.monitor,
#                                       square_size=protocol['square-size (deg)'],
#                                       noise_mean_refresh_time=protocol['mean-refresh-time (s)'],
#                                       noise_rdm_jitter_refresh_time=protocol['jitter-refresh-time (s)'],
#                                       seed=protocol['noise-seed (#)'])

#         self.experiment = {'refresh-times':self.STIM['t']}
            

if __name__=='__main__':

    protocol = {'Stimulus':'light-level'}
    
    stim = visual_stim_replay(None, protocol)
    stim.get_frame(0)

        
