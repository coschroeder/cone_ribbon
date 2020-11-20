import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from IPython.display import display

import ipywidgets as widgets
from ipywidgets import (interact, interactive )


# import model
import sys
sys.path.append('../model_implementation/')
from model_new2020 import solve_ribbon_ode


"""
setting up a simplified model
"""

def get_all_params(RRP_size, IP_size, max_release):
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
    standardized_params[4] = 0.32 # np.mean(params_unnorm_mean[:,4])=0.317
    
    # normalize it
    #boundsnorm = [[0, 5], [0, 5], [0, 20], [2, 30], [0, 1], [-0.5, 10], [-0.5, 5]]
    #params_standardized_norm = normalize_vec(standardized_params, boundsnorm)
    
    return standardized_params

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
define stimulus
"""

def get_flash_stim(len_flash, len_low, len_high, max_amp,len_adapt, amp_adapt, dt=0.032):
    """
    all values in sec
    len_tot: total time without adaptation time (which is given in negative times)
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
        x[bound0:bound1] = max_amp
        i+=1
              
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
                                      max=8.0,
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
                                                 ('Noise', 4)],
                                            value=1,
                                            description='Stimulus:',
                                        style=style,
                                        #layout=Layout
                                        )
    
    stim_freq_slider = widgets.FloatSlider(value=0.5,
                                      min=0.05,
                                      max=1,
                                    step=0.05,
                                    description='Stimulus Frequency:',
                                    disabled=False,
                                    continuous_update=False,
                                            style=style)
    
    
    # track plot
    trackplot_checkbox = widgets.Checkbox(value=False,
                        description='Track changes',
                        disabled=False)
    
    # clear plot button
    clearplot_button = widgets.Button(description='Clear plot',
                                        disabled=False,
                                        button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Clears the figure for the next plotting event.',
                                        icon='' #check (FontAwesome names without the `fa-` prefix)
                                        )




    
    return RRP_slider, IP_slider, max_release_slider, stimulus_dropdown,stim_freq_slider, trackplot_checkbox, clearplot_button

    
def get_stimulus_choice(stimulus_mode, freq):
     # choose stimulus
        if stimulus_mode==1:
            # flash stimulus "original as in paper"
            len_flash = 58#  sec
            max_amp= 1
            len_low = 3 # sec
            len_high = 3 # sec 
            len_adapt = 10
            amp_adapt = 0.2
            stimulus,t =  get_flash_stim(len_flash, len_low, len_high, max_amp, len_adapt, amp_adapt,)
            stimulus = normalize_specific(stimulus)

        elif stimulus_mode==2:
            # High frequency
            len_flash = 58#  sec
            max_amp= 1
            len_low = 1/freq # sec
            len_high = 1/freq # sec 
            len_adapt = 10
            amp_adapt = 0.2
            stimulus,t =  get_flash_stim(len_flash, len_low, len_high, max_amp, len_adapt, amp_adapt,)
            stimulus = normalize_specific(stimulus)


        elif stimulus_mode==3:
            # sine stimulus
            len_stim = 58 #sec
            f = freq # frequency
            len_adapt = 10
            amp_adapt = -0.5 # will be normalized later 

            t = np.arange(0,len_stim,0.032)
            stimulus = np.sin(2*np.pi*(t-len_adapt) * f)
            stimulus[t<len_adapt] = amp_adapt
            

        elif stimulus_mode==4:
            # noise stimulus
            len_stim = 58 #sec
            tpts_per_value = int((1/freq) / 0.032) 
            len_adapt = 10
            amp_adapt = 0.2

            t = np.arange(0,len_stim,0.032)
            stimulus = produce_white_noise(len(t),steplen=tpts_per_value)
            stimulus[t<len_adapt] = amp_adapt
        
        return stimulus,t
    
    
    
    
class Ribbon_Plot():
    def __init__(self):
        backend_ =  mpl.get_backend() 
        mpl.use("Agg")  # Prevent showing stuff
        self.set_new_fig()
        mpl.use(backend_) # Reset backend        
        self.i = 0
        
    def plot_ribbon(self, RRP_size, IP_size, max_release, stimulus_mode,freq, track_changes=False):
       
        # get stimulus
        stimulus,t = get_stimulus_choice(stimulus_mode, freq)

        # get all parameters
        params_standardized = get_all_params(RRP_size, IP_size, max_release)
        # run simulation
        simulation = solve_ribbon_ode(stimulus, *params_standardized)

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
        norm = mpl.colors.Normalize(vmin=0, vmax=5) # 5 values
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
        cmap.set_array([])
        color=cmap.to_rgba(self.i)
        # simulation
        self.fig1.axes[0].plot(t,simulation, color=color)

        # stimulus
        stimulus -= np.min(stimulus)
        stimulus /= np.max(stimulus)
        self.fig1.axes[1].plot(t,stimulus, color=color)
        
        
        if track_changes and self.i>1:
            display(self.fig1)
            
        
            
            
    def set_new_fig(self):
 
        self.fig1 = plt.figure(1, figsize=(8,6))
        ax1 = plt.subplot(211)
        self.fig1.add_axes(ax1)
        ax1.set_ylim(-0.1,5)
        self.fig1.axes[0].set_xticklabels([])
        ax1.set_ylabel('Glutamate Release Rate \n [ves.u./sec.]')


        ax2 = plt.subplot(212)
        self.fig1.add_axes(ax2)
        ax2.set_xlabel('sec')
        ax2.set_ylabel('"Stimulus" \n (Ca Concentration) [a.u.]')
        #self.fig1.axes[1].text(70,0.2,'Step duration: \n%.2f sec'%len_high)
        sns.despine()
        


    def clearplot_button_click(self,b):
        backend_ =  mpl.get_backend() 
        mpl.use("Agg")  # Prevent showing stuff
        self.set_new_fig()
        mpl.use(backend_) # Reset backend

    
    
    def plot_interactive_ribbon(self):
        
        RRP_slider, IP_slider, max_release_slider, stimulus_dropdown,stim_freq_slider,trackplot_checkbox, clearplot_button = get_sliders()
    
        # create interactive plot

        plot_widgets = interactive(self.plot_ribbon, 
                           RRP_size = RRP_slider, 
                           IP_size = IP_slider, 
                           max_release = max_release_slider, 
                           stimulus_mode=stimulus_dropdown,
                            freq = stim_freq_slider,
                          track_changes = trackplot_checkbox)



        # reshape layout. use the interactive function instead of interact
        controls = widgets.HBox(plot_widgets.children[:-1], layout = widgets.Layout(flex_flow='row wrap'))
        output = plot_widgets.children[-1]
        display(widgets.VBox([controls, output, clearplot_button]))
               
        clearplot_button.on_click(self.clearplot_button_click)

    