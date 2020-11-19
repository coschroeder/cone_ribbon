import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import ipywidgets as widgets
from ipywidgets import interact, interactive


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
    stimulus_dropdown = widgets.Dropdown(options=[('Step Stimulus', 1), 
                                                   ('Short Steps', 2), 
                                                   ('Long Steps', 3)],
                                            value=1,
                                            description='Stimulus:',
                                        style=style,
                                        #layout=Layout
                                        )
    
    return RRP_slider, IP_slider, max_release_slider, stimulus_dropdown



def plot_ribbon(RRP_size, IP_size, max_release, stimulus_mode):
    
    # choose stimulus
    if stimulus_mode==1:
        # flash stimulus
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
        len_low = 1 # sec
        len_high = 1 # sec 
        len_adapt = 10
        amp_adapt = 0.2
        stimulus,t =  get_flash_stim(len_flash, len_low, len_high, max_amp, len_adapt, amp_adapt,)
        stimulus = normalize_specific(stimulus)
        
        
    elif stimulus_mode==3:
        # long step
        len_flash = 58#  sec
        max_amp= 1
        len_low = 10 # sec
        len_high = 10 # sec 
        len_adapt = 10
        amp_adapt = 0.2
        stimulus,t =  get_flash_stim(len_flash, len_low, len_high, max_amp, len_adapt, amp_adapt,)
        stimulus = normalize_specific(stimulus)
        
    
    # get all parameters
    params_standardized = get_all_params(RRP_size, IP_size, max_release)
    # run simulation
    simulation = solve_ribbon_ode(stimulus, *params_standardized)
    
    
    # plotting
    sns.set_context('notebook')
    plt.figure(1, figsize=(8,6))
    
    # simulation
    ax = plt.subplot(211)
    plt.plot(t,simulation)
    plt.ylim(-0.1,5)
    ax.set_xticklabels([])
    plt.ylabel('Glutamate Release Rate \n [ves.u./sec.]')
    

    
    # stimulus
    plt.subplot(212)
    stimulus -= np.min(stimulus)
    stimulus /= np.max(stimulus)
    plt.plot(t,stimulus)
    plt.xlabel('sec')
    plt.ylabel('"Stimulus" \n (Ca Concentration) [a.u.]')
    plt.text(70,0.2,'Step duration: \n%.2f sec'%len_high)
    sns.despine()
    
    
    
    
def plot_interactive_ribbon():
    # get the necessary predefined sliders
    RRP_slider, IP_slider, max_release_slider, stimulus_dropdown = get_sliders()
    
    # create interactive plot
    
    widgets = interact(plot_ribbon, 
                       RRP_size = RRP_slider, 
                       IP_size = IP_slider, 
                       max_release = max_release_slider, 
                       stimulus_mode=stimulus_dropdown)
    
    
    
    # reshape layout. use the interactive function instead of interact
    #controls = HBox(widgets.children[:-1], layout = Layout(flex_flow='row wrap'))
    #output = widgets.children[-1]
    #display(VBox([controls, output]))