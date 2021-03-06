import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from IPython.display import display
from IPython.display import clear_output

import ipywidgets as widgets
from ipywidgets import (interact, interactive )

from ipywidgets import GridspecLayout



# import model
import sys
sys.path.append('../model_implementation/')
from model_new2020 import solve_ribbon_ode


"""
setting up a simplified model
"""

def get_all_params(RRP_size, IP_size, max_release, ca_baseline):
    """
    returns all 7 paramters for the ribbon model
    calculates the missing changing rates based on the pool sizes.
    non-linearity is fixed here.
    ---
    these are educated hand picked parameter combination guesses
    """
        
    standardized_params = np.zeros(7)

    # RRP_size
    standardized_params[6] = RRP_size
    # max release rate
    standardized_params[2] = max_release*RRP_size #0.6*RRP_size
    # RRP refill rate
    standardized_params[1] = 0.4*IP_size

    # IP_size 
    standardized_params[5] = IP_size
    # IP refill rate
    standardized_params[0] = 0.2*IP_size

    #k and x0: mean over the three modes of moG
    standardized_params[3] = 10.2 # np.mean(params_unnorm_mean[:,3])=10.23
    standardized_params[4] = ca_baseline #0.32 # np.mean(params_unnorm_mean[:,4])=0.317
    
    # normalize it
    #boundsnorm = [[0, 5], [0, 5], [0, 20], [2, 30], [0, 1], [-0.5, 10], [-0.5, 5]]
    #params_standardized_norm = normalize_vec(standardized_params, boundsnorm)
    
    return standardized_params

# define zone specific parameters
def get_zone_params(zone):
    """
    for zone fits (mean of distributions)
    zone in ['AZ','N','D']
    returns: unnormalized params for RRP_size, IP_size, max_release, ca_baseline
    """
    if zone=='AZ':
        RRP_size = 5 # rounded(4.9)
        IP_size = 7.8 # rounded(7.7)
        max_release = 0.55 # 2.6 = round(0.55*RRP_size)
        ca_baseline = 0.2 #0.32 +0.08 (shifted ca) 

    elif zone=='N':
        RRP_size = 1.0 # rounded(1.04)
        IP_size = 10.8 # rounded(10.72)
        max_release = 0.95  # 10.1 = round(0.95*RRP_size)
        ca_baseline = 0.5 #0.32 +0.08 (shifted ca) 

    elif zone=='D':
        RRP_size = 3 #rounded(2.9) 
        IP_size = 7 # rounded(7.08) 
        max_release = 0.65 # 2.0 = round(0.65*RRP_size)
        ca_baseline = 0.25 #0.32 +0.08 (shifted ca) 
        
    return RRP_size, IP_size, max_release, ca_baseline

# additional scaling of input stimulus 
# st. the baseline of the non-linearity level matches the Ca level in the data
def normalize_specific(data, target_max=0.4, target_min=-0.08):
    """
    normalize the data to same max and min values as target
    """
    data = np.copy(data)
    data /= (np.max(data)-np.min(data))
    data *= (target_max - target_min)
    data += target_min - np.min(data)
    return data

"""
defining addition stim to ca kernel
"""

def cone_kernel(scale=1, dt=0.032):
    """
    PR biphasic kernel
    :param scale: time scale 
    :return: normalized kernel with dt=1ms
    # adapted from https://github.com/coschroeder/abc-ribbon/blob/master/standalone_model/ribbonv2.py
    """
    # all variables in sec
    #dt = 0.032
    t = np.arange(0,0.3,dt)

    tau_r = 0.07 * scale 
    tau_d = 0.07 *scale 
    phi = -np.pi * 0.2 * scale 
    tau_phase = 100
    kernel = -(t/tau_r)**3 / (1+t/tau_r) * np.exp(-(t/tau_d)**2) * np.cos(2*np.pi*t / phi + tau_phase)
    return kernel/ np.abs(np.sum(kernel)) 

def ca_to_ca_kernel(tau_decay , dt=0.032):
    """
    dt in sec
    tau_decay default: 1.5 [sec]
    """
    t = np.arange(0,3,dt)
    
    tau_rise = 0.03 #sec
    #tau_decay = 2*0.755 # radius dependent: radius*0.755 sec/um
    kernel = (1-np.exp(-t/(tau_rise)) )* np.exp(-t/tau_decay)
    
    return kernel/np.sum(kernel), t


def ca_simulation(stimulus_raw, tau_decay,  dt=0.032):
    
    # add tpts for more stable convolution
    tpts_to_add = int(5/dt) #corresponding to 5 sec
    stimulus = np.hstack([np.ones(tpts_to_add)*stimulus_raw[0],stimulus_raw])
    
    # get kernels
    ca_kernel1 = cone_kernel(dt=dt)
    ca_kernel2, t_kernel = ca_to_ca_kernel(tau_decay,dt=dt)
    
    # process "receptore impulse response"
    ca1 = np.convolve(ca_kernel1, stimulus,mode='full')[:len(stimulus)]
    # non-linearity
    ca1= np.exp(3*ca1)
    
    # ca current to ca concentration
    ca2 = np.convolve(ca_kernel2, ca1,mode='full')[:len(stimulus)]
    
    return ca2[tpts_to_add:len(stimulus_raw)+tpts_to_add]


"""
define stimulus
"""

def get_flash_stim(len_flash, len_low, len_high, max_amp,len_adapt, amp_adapt, dt=0.032):
    """
    all values in sec
    """
    t = np.arange(len_adapt, len_flash+len_adapt, dt) # in sec
    x = np.zeros(len(t))
    
    
    low_tpts =int(len_low/dt)
    high_tpts =int(len_high/dt)
    
    bound1=0
    i=0
    while bound1<len(t):
        bound0 = low_tpts*i + high_tpts*i #+ low_tpts
        bound1 = low_tpts*i + high_tpts*(i+1) #+ low_tpts
        x[bound0+low_tpts:bound1+low_tpts] = max_amp
        i+=1
              
    # stack adatation time
    t2 = np.arange(0,len_adapt, dt)
    t = np.hstack([t2,  t])
    x = np.hstack([np.ones(len(t2))*amp_adapt,  x])

    return x, t



def get_periodic_flash_stim(len_flash, freq, max_amp, len_adapt, amp_adapt, dt=0.032):
    """
    all values in sec
    """
    t = np.arange(len_adapt, len_flash+len_adapt, dt) # in sec
    x = np.sin(2*np.pi*(t-len_adapt)*freq)
    
    x[x>=0]=0
    x[x<0]=1
              
    # stack adatation time
    t2 = np.arange(0,len_adapt, dt)
    t = np.hstack([t2,  t])
    x = np.hstack([np.ones(len(t2))*amp_adapt,  x])

    return x, t


def produce_white_noise(tpts, std=0.1, seed=1234, steplen=1, dist='uniform'):
    """
    std: std of noise for normal dist
    tpts: len of noise
    ---
    dist: uniform or normal
    """
    
    #produce white noise stim
    
    np.random.seed(seed)
    
    if steplen==1:
        if dist=='uniform':
            white_noise = np.random.uniform(size= tpts)
        elif dist=='normal':
            white_noise = np.random.normal(0,std,sizetpts)

    else:
        white_noise = np.zeros(tpts)
        for i in range(int(len(white_noise)/steplen)):
            if dist=='uniform':
                white_noise[i*steplen:i*steplen+steplen] = np.random.uniform()
            elif dist=='normal':
                white_noise[i*steplen:i*steplen+steplen] = np.random.normal(0, std)
            
    return white_noise



"""
setting up interactive plotting functions
"""


def get_sliders():
    # define sliders

    style = {'description_width': 'initial'} 
    # layout=Layout(width='50%')

    RRP_slider = widgets.FloatSlider(value=4,
                                      min=0.2,
                                      max=10.0,
                                      step=0.2,
                                      description='RRP Size:',
                                    disabled=False,
                                    continuous_update=False, 
                                    style = style
                                    )

    IP_slider = widgets.FloatSlider(value=10, 
                                      min=0.2,
                                      max=20.0,
                                      step=0.2,
                                    description='IP Size:',
                                    disabled=False,
                                    continuous_update=False,
                                    style = style
                                   )

    max_release_slider = widgets.FloatSlider(value=0.5,
                                      min=0.05,
                                      max=1,
                                    step=0.05,
                                    description='Max. Release Rate:',
                                    disabled=False,
                                    continuous_update=False,
                                            style=style)

    # stimulus selection
    stimulus_dropdown = widgets.Dropdown(options=[('Fixed Step', 1), 
                                                   ('Steps', 2), 
                                                    ('Sine', 3),
                                                 ('Noise', 4),
                                                 ('Chirp',5)],
                                            value=1,
                                            description='Stimulus:',
                                        style=style,
                                        #layout=Layout
                                        )
    
    stim_freq_slider = widgets.FloatSlider(value=0.6,
                                      min=0.1,
                                      max=10,
                                    step=0.1,
                                    description='Stimulus Frequency [Hz]:',
                                    disabled=False,
                                    continuous_update=False,
                                            style=style)
    
    
    tau_decay_slider = widgets.FloatSlider(value=.5,
                                      min=0.05,
                                      max=3,
                                    step=0.05,
                                    description='Ca kernel (tau):',
                                    disabled=False,
                                    continuous_update=False,
                                            style=style)
    
    ca_baseline_slider = widgets.FloatSlider(value=.3,
                                      min=-0.2,
                                      max=1.3,
                                    step=0.05,
                                    description='Ca offset [a.u.]:',
                                    disabled=False,
                                    continuous_update=False,
                                            style=style)
    
    
    # track plot
    trackplot_checkbox = widgets.Checkbox(value=False,
                        description='Track changes',
                        disabled=False)
    
    time_resolution_ms_slider = widgets.IntSlider(value=32,
                                      min=2,
                                      max=32,
                                    step=2,
                                    description='Time resulution [ms]:',
                                    disabled=False,
                                    continuous_update=False,
                                            style=style)
    
    # execute plot button
    execute_button = widgets.Button(description='Execute.',
                                        disabled=False,
                                        button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Executes the simulation the first time.',
                                        icon='' #check (FontAwesome names without the `fa-` prefix)
                                        )
    # clear plot button
    clearplot_button = widgets.Button(description='Clear plot',
                                        disabled=False,
                                        button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Clears the figure for the next plotting event.',
                                        icon='' #check (FontAwesome names without the `fa-` prefix)
                                        )
    
    # define slider for zone specific values
    az_button = widgets.Button(description='Set to AZ values',
                                        disabled=False,
                                        button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Set the slider values to AZ fits.',
                                        icon='' #check (FontAwesome names without the `fa-` prefix)
                                        )
        
    nz_button = widgets.Button(description='Set to N values',
                                        disabled=False,
                                        button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Set the slider values to nasal fits.',
                                        icon='' #check (FontAwesome names without the `fa-` prefix)
                                        )
    dz_button = widgets.Button(description='Set to D values',
                                        disabled=False,
                                        button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Set the slider values to dorsal zone fits.',
                                        icon='' #check (FontAwesome names without the `fa-` prefix)
                                        )
    





    
    return (RRP_slider, IP_slider, max_release_slider, 
            stimulus_dropdown, stim_freq_slider, 
            trackplot_checkbox, tau_decay_slider, time_resolution_ms_slider, 
            az_button, nz_button, dz_button,
            execute_button, clearplot_button, ca_baseline_slider)

    
def get_stimulus_choice(stimulus_mode, freq, dt):
    """
    stimulus_mode in [1,...,5] corresponding to [original_flashes, flashes, sine, noise, chirp]
    """
     # choose stimulus
    if stimulus_mode==1:
        # flash stimulus "original as in paper"
        len_flash = 60#  sec
        max_amp= 1
        freq=1/6
        len_adapt = 13.6
        amp_adapt = 0.5
        stimulus,t =  get_periodic_flash_stim(len_flash, freq, max_amp, len_adapt, amp_adapt,dt)

    elif stimulus_mode==2:
        # flashes
        len_flash = 58#  sec
        max_amp= 1
        len_adapt = 10
        amp_adapt = 0.5
        stimulus,t =  get_periodic_flash_stim(len_flash, freq, max_amp, len_adapt, amp_adapt,dt)


    elif stimulus_mode==3:
        # sine stimulus
        len_stim = 58 #sec
        f = freq # frequency
        len_adapt = 10
        amp_adapt = 0 # will be normalized later 

        t = np.arange(0,len_stim,dt)
        stimulus = -np.sin(2*np.pi*(t-len_adapt) * f)
        stimulus[t<len_adapt] = amp_adapt


    elif stimulus_mode==4:
        # noise stimulus
        len_stim = 58 #sec
        tpts_per_value = int((1/freq) / dt) 
        len_adapt = 10
        amp_adapt = 0.5

        t = np.arange(0,len_stim,dt)
        stimulus = produce_white_noise(len(t),steplen=tpts_per_value)
        stimulus[t<len_adapt] = amp_adapt

    elif stimulus_mode==5:
        # accelerating sine: chirp
        len_stim = 58 #sec
        f_start = 0.1  # frequency to start
        f_end = 5

        len_adapt = 10
        amp_adapt = 0 # will be normalized later 

        t = np.arange(0,len_stim,dt)
        f = np.linspace(f_start,f_end,num=len(t[t>len_adapt]))
        stimulus = np.zeros(len(t))
        stimulus[t>len_adapt] = -np.sin(2*np.pi*(t[t>len_adapt]-len_adapt) * f)
        stimulus[t<len_adapt] = amp_adapt

    return stimulus,t
    
"""
plot of a ribbon comic
"""
def plot_ribbon_schema(ax, n_RRP,n_IP, titlesize):
    """
    plots a comic of the ribbon
    """
    ax.clear()

    ax.set_title('Ribbon schema',fontsize=titlesize)
    #sns.set_context('talk')
    
    # clear axes
    
    r=0.1 #radius of vesicles
    ves_dist = 2.4 #  distance between vesicles (*r)
    
    # compute height and width
    # assume constant vesicle densities at the ribbon
    
    # set to int
    n_IP /=2
    n_IP = max(1,int(np.round(n_IP)))
    n_RRP = max(1,int(np.round(n_RRP)))

    width = (n_RRP-1)*r*ves_dist 
    height= n_IP*r*ves_dist
    
    # set up figure
    #fig=plt.figure(1,figsize=(6,6))
    #ax = plt.gca()
   
    # keep plot quadratic  
    ax.set_xlim(-2,4)
    ax.set_ylim(0,4)

    

    ### plot ribbon
    ribboncolor = 'goldenrod'
    xy=(0,0.1+2*r)
    
    box= mpl.patches.Rectangle(xy,width,height,
                               joinstyle= 'round',
                               fill=True,
                               #color='gold',
                               edgecolor='black',
                               lw=1,
                               facecolor=ribboncolor
                              )
    ax.add_patch(box)

    
    ## vesicles
    vesicles_args = dict(edgecolor='black', 
                         lw=2,
                         radius=r)

    # add vesicles IP
    for i in range(n_IP):
        xyv=(xy[0]-0.1, xy[1]+(i+1)*(ves_dist*r))
        v = mpl.patches.Circle(xyv,facecolor='white', **vesicles_args)
        ax.add_patch(v)
        
    for i in range(n_IP):
        xyv=(xy[0]+width+0.1, xy[1]+(i+1)*(ves_dist*r))
        v = mpl.patches.Circle(xyv,facecolor='white',**vesicles_args)
        ax.add_patch(v)


    # add vesicles RRP
    for i in range(n_RRP):
        xyv=(xy[0]+i*r*ves_dist, xy[1]-r)
        v = mpl.patches.Circle(xyv,facecolor='grey', **vesicles_args)
        ax.add_patch(v)


    # plot membrane 
    ax.axhline(0, color='black',lw=3)

    # add text 
    ax.text(-2,-0.5,'Membrane')
    ax.text(xy[0],height+0.5,'Ribbon',color=ribboncolor)
    ax.text(width+4*r,0.15,'RRP',color='grey')
    ax.text(width+4*r,height+1*r,'IP',color='black')

    ax.axis('off')
    #return fig

def plot_ca_kernel(ax, tau_decay,titlesize):
    ax.clear()
    ax.set_title('Ca kernel', fontsize=titlesize)
    ax.set_xlabel('sec')
    ax.set_ylabel('a.u.')
    ax.set_yticks([0,1])
    ax.set_xticks([0,1,2,3])
    ca_kernel, t_kernel = ca_to_ca_kernel(tau_decay,dt=0.001)
    ax.plot(t_kernel,ca_kernel/np.max(ca_kernel), color='red')       

    
    
class Ribbon_Plot():
    def __init__(self, figsize=(30,8)):
        self.figsize=figsize
        self.titlesize=15
        self.i = 0
        self.dt=0.032
        # set up initial figure
        backend_ =  mpl.get_backend() 
        mpl.use("Agg")  # Prevent showing stuff
        self.set_new_fig()
        mpl.use(backend_) # Reset backend    

        # set sliders
        (self.RRP_slider, 
         self.IP_slider, 
         self.max_release_slider, 
         self.stimulus_dropdown,
         self.stim_freq_slider,
         self.trackplot_checkbox,
         self.tau_decay_slider,
         self.time_resolution_ms_slider,
         self.az_button,
         self.nz_button,
         self.dz_button,
         self.execute_button,
         self.clearplot_button,
         self.ca_baseline_slider) = get_sliders()
        
        self.update = True # controls if figure is updated or not
        self.plot_interactive_ribbon()
        
        '''
        # plot first
        backend_ =  mpl.get_backend() 
        mpl.use("Agg")  # Prevent showing stuff
        self.plot_ribbon(RRP_size = self.RRP_slider.value, 
                        IP_size = self.IP_slider.value, 
                        max_release = self.max_release_slider.value, 
                        stimulus_mode = self.stimulus_dropdown.value,
                        freq = self.stim_freq_slider.value,
                        tau_decay = self.tau_decay_slider.value,
                        time_resolution_ms = self.time_resolution_ms_slider.value,
                        track_changes = self.trackplot_checkbox.value)
        mpl.use(backend_) # Reset backend
        #display(self.fig1)
        '''

        
        
    def set_new_fig(self):
        layout = (5,6) #nrows, ncolumns
            
        self.fig1 = plt.figure(1, figsize=self.figsize)
        
        # ax0-2 simulation
        ax0 = plt.subplot2grid(layout,(0,0), rowspan=3,colspan=4)
        ylims_glut = (-0.1,5)
        self.fig1.add_axes(ax0)
        ax0.set_ylim(ylims_glut)
        self.fig1.axes[0].set_xticklabels([])
        ax0.set_ylabel('Glutamate Release Rate \n [ves.u./sec.]')
        ax0.set_title('Simulation of Vesicle Release', fontsize=self.titlesize)
        sns.despine(ax=ax0)

        ax1 = plt.subplot2grid(layout,(3,0), rowspan=1,colspan=4)
        ax1.set_xticklabels([])
        ax1.set_ylabel('Ca Concentration \n [a.u.]')
        self.fig1.add_axes(ax1)
        sns.despine(ax=ax1)

        
        ax2 = plt.subplot2grid(layout,(4,0), rowspan=1,colspan=4)
        self.fig1.add_axes(ax2)
        ax2.set_xlabel('sec')
        ax2.set_ylabel('Stimulus \n [normalized]')
        sns.despine(ax=ax2)
        
        
        # for zoom in 
        ax3 = plt.subplot2grid(layout,(0,4), rowspan=3,colspan=1)
        ax3.set_title('Zoom in', fontsize=self.titlesize)
        ax3.set_ylim(ylims_glut)
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_yticks([])
        sns.despine(ax=ax3,left=True)

        self.fig1.add_axes(ax3)
        ax4 = plt.subplot2grid(layout,(3,4), rowspan=1,colspan=1)
        ax4.set_xticklabels([])
        ax4.set_yticklabels([])
        ax4.set_yticks([])
        sns.despine(ax=ax4, left=True)

        
        self.fig1.add_axes(ax4)
        ax5 = plt.subplot2grid(layout,(4,4), rowspan=1,colspan=1)
        ax5.set_xlabel('sec')
        ax5.set_yticklabels([])
        ax5.set_yticks([])
        self.fig1.add_axes(ax5)
        sns.despine(ax=ax5,left=True)

        
        # ribbon comic
        ax6 = plt.subplot2grid(layout,(0,5), rowspan=2,colspan=1)
        #ax6.set_title('Ribbon schema', fontsize=self.titlesize)
        self.fig1.add_axes(ax6)
        #ax3.set_xlabel('sec')
        #ax3.set_ylabel('Stimulus \n [normalized]')
        
        # for text
        ax7 = plt.subplot2grid(layout,(2,5), rowspan=1,colspan=1)
        self.fig1.add_axes(ax7)
        
        # for Ca kernel
        ax8 = plt.subplot2grid(layout,(4,5), rowspan=1,colspan=1)
        self.fig1.add_axes(ax8)
        sns.despine(ax=ax8)

        
        #sns.despine()
        plt.tight_layout()        
        
        
        
    def plot_ribbon(self, RRP_size, IP_size, max_release, stimulus_mode,freq,tau_decay,time_resolution_ms, ca_baseline, track_changes=False):
        
        if self.update:
            backend_ =  mpl.get_backend() 
            mpl.use("Agg")  # Prevent showing stuff
            self.dt = time_resolution_ms/1000 # change to sec

            # get stimulus
            stimulus,t = get_stimulus_choice(stimulus_mode, freq,dt=self.dt)

            # simulate calcium concentration
            ca_concentration = ca_simulation(stimulus, tau_decay,dt=self.dt)
            ca_concentration_norm = normalize_specific(ca_concentration, target_max=1, target_min=0) 


            # get all parameters
            params_standardized = get_all_params(RRP_size, IP_size, max_release,ca_baseline)
            # run simulation
            simulation = solve_ribbon_ode(ca_concentration_norm, *params_standardized, dt=self.dt)

            # plotting
            sns.set_context('notebook')

            # set up figure
            if not track_changes:
                self.set_new_fig()
                self.i=5
            else:
                if self.i>=5:
                    self.set_new_fig()
                    self.i=0
                self.i+=1

            # actual plotting:
            # set color
            norm = mpl.colors.Normalize(vmin=-3, vmax=5) # 7 values but use only 5
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
            cmap.set_array([])
            color=cmap.to_rgba(self.i)
            # simulation
            self.fig1.axes[0].plot(t,simulation, color=color)

            # Ca 
            self.fig1.axes[1].plot(t,ca_concentration, color=color)

            # stimulus
            stimulus -= np.min(stimulus)
            stimulus /= np.max(stimulus)
            self.fig1.axes[2].plot(t,stimulus, color=color)


            # plot zoom ins
            xlims = (19.4,25.6)# (19.6,25.6)
            # simulation
            self.fig1.axes[3].plot(t,simulation, color=color)
            self.fig1.axes[3].set_xlim(xlims)

            # Ca 
            self.fig1.axes[4].plot(t,ca_concentration, color=color)
            self.fig1.axes[4].set_xlim(xlims)

            # stimulus
            self.fig1.axes[5].plot(t,stimulus, color=color)
            self.fig1.axes[5].set_xlim(xlims)

            # plot ribbon schema
            plot_ribbon_schema(self.fig1.axes[6],RRP_size,IP_size,titlesize=self.titlesize)

            # plot some text
            self.fig1.axes[7].text(0,-2, 'This is a simplified \nribbon schema \nwhich assumes constant \nvesicle density \nat the ribbon. \nThis is not necessarily \nthe case.')
            self.fig1.axes[7].axis('off')


            # plot Ca kernel
            plot_ca_kernel(self.fig1.axes[8], tau_decay, self.titlesize)


            mpl.use(backend_) # Reset backend  
            display(self.fig1) 
        
        # this is used for setting the zone buttons. only one plotting event for multiple slider changes
        else:
            pass

    def clearplot_button_click(self,b):
        backend_ =  mpl.get_backend() 
        mpl.use("Agg")  # Prevent showing stuff
        self.set_new_fig()
        mpl.use(backend_) # Reset backend
        self.i=0
        
    def execute_button_click(self,b):
        backend_ =  mpl.get_backend() 
        mpl.use("Agg")  # Prevent showing stuff

        self.plot_ribbon(RRP_size = self.RRP_slider.value, 
                        IP_size = self.IP_slider.value, 
                        max_release = self.max_release_slider.value, 
                        stimulus_mode = self.stimulus_dropdown.value,
                        freq = self.stim_freq_slider.value,
                        tau_decay = self.tau_decay_slider.value,
                        time_resolution_ms = self.time_resolution_ms_slider.value,
                        track_changes = self.trackplot_checkbox.value)
                
        mpl.use(backend_) # Reset backend
        self.i=0
        #display(self.fig1)
    
    # specify zone buttons
    def set_az_values(self, b):
        RRP_size, IP_size, max_release, ca_baseline = get_zone_params('AZ')
        self.update=False
        self.RRP_slider.value = RRP_size
        self.IP_slider.value = IP_size
        self.max_release_slider.value = max_release
        self.ca_baseline_slider.value = ca_baseline
        self.tau_decay_slider.value = 0.001 # hack to make sure that there is a slider change for the last slider update
        self.update=True
        self.tau_decay_slider.value = 0.45
        
        
    def set_nz_values(self, b):
        RRP_size, IP_size, max_release,ca_baseline = get_zone_params('N')
        self.update=False
        self.RRP_slider.value = RRP_size
        self.IP_slider.value = IP_size
        self.max_release_slider.value =max_release
        self.ca_baseline_slider.value = ca_baseline
        self.tau_decay_slider.value = 0.001 # hack to make sure that there is a slider change for the last slider update
        self.update=True
        self.tau_decay_slider.value = 0.45
        
    def set_dz_values(self, b):
        RRP_size, IP_size, max_release, ca_baseline = get_zone_params('D')
        self.update=False

        self.RRP_slider.value = RRP_size
        self.IP_slider.value = IP_size
        self.max_release_slider.value = max_release
        self.ca_baseline_slider.value = ca_baseline
        self.tau_decay_slider.value = 0.001 # hack to make sure that there is a slider change for the last slider update
        self.update=True
        self.tau_decay_slider.value = 0.45  


    def plot_interactive_ribbon(self, layout='h'):
       
        # create interactive plot

        plot_widgets = interactive(self.plot_ribbon, 
                                    RRP_size = self.RRP_slider, 
                                    IP_size = self.IP_slider, 
                                    max_release = self.max_release_slider, 
                                    stimulus_mode = self.stimulus_dropdown,
                                    freq = self.stim_freq_slider,
                                    tau_decay = self.tau_decay_slider,
                                    time_resolution_ms = self.time_resolution_ms_slider,
                                    track_changes = self.trackplot_checkbox,
                                    ca_baseline = self.ca_baseline_slider)


        if layout=='v':
            # reshape layout. use the interactive function instead of interact

            self.grid = GridspecLayout(5, 3)
            # ribbon 
            self.grid[0, 0] = plot_widgets.children[0]
            self.grid[1, 0] = plot_widgets.children[1]
            self.grid[2, 0] = plot_widgets.children[2]
            self.grid[3, 0] = plot_widgets.children[7] # ca baseline

            # stimulus 
            self.grid[0, 1] = plot_widgets.children[3] 
            self.grid[1, 1] = plot_widgets.children[4]
            self.grid[2, 1] = widgets.Label(value="Frequency slider is only valid for certain stimuli.")

            # rest 
            self.grid[0, 2] = plot_widgets.children[5] # ca kernel
            self.grid[2, 2] = plot_widgets.children[6] # time res
            self.grid[3, 2] = plot_widgets.children[8] # track changes

            # set zones
            self.grid[4, 0] = self.az_button
            self.grid[4, 1] = self.nz_button
            self.grid[4, 2] = self.dz_button

            self.output = plot_widgets.children[-1]
        
            output_boxed = widgets.VBox([self.grid,self.output,self.clearplot_button])
            
        elif layout=='h':
            # reshape layout. use the interactive function instead of interact

            self.grid = GridspecLayout(14, 1)
            # ribbon 
            self.grid[3, 0] = plot_widgets.children[0]
            self.grid[4, 0] = plot_widgets.children[1]
            self.grid[5, 0] = plot_widgets.children[2]
            self.grid[6, 0] = plot_widgets.children[7] # ca baseline

            # stimulus 
            self.grid[0, 0] = plot_widgets.children[3] 
            self.grid[1, 0] = plot_widgets.children[4]
            self.grid[2, 0] = widgets.Label(value="Frequency slider is only valid for certain stimuli.")

            # rest 
            self.grid[7, 0] = plot_widgets.children[5] # ca kernel
            self.grid[8, 0] = plot_widgets.children[6] # time res
            self.grid[9, 0] = plot_widgets.children[8] # track changes

            # set zones
            self.grid[10, 0] = self.az_button
            self.grid[11, 0] = self.nz_button
            self.grid[12, 0] = self.dz_button
            
            # clear plot
            self.grid[13, 0] = self.clearplot_button

            self.output = plot_widgets.children[-1]
        
            output_boxed = widgets.HBox([self.grid,self.output])
        
        # set handelrs for buttons
        self.execute_button.on_click(self.execute_button_click)
        self.clearplot_button.on_click(self.clearplot_button_click)

        # set handlers for zone buttons
        self.az_button.on_click(self.set_az_values)
        self.nz_button.on_click(self.set_nz_values)
        self.dz_button.on_click(self.set_dz_values)

            
        display(output_boxed)
        
        return output_boxed


    