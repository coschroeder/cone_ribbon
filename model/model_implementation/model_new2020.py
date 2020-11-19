import numpy as np
import scipy as scp


"""
change ODE solver RK23 to have minimal stepsize
"""
from scipy.integrate._ivp.rk import *

class MyRK23(RK23):
    def __init__(self, *args, min_step_hard, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_step_hard = min_step_hard #kwargs['min_step_hard']
        
    def _step_impl(self):
        """
        changed to have hard minimla stepsize, ignoring tolerances.
        from: https://github.com/scipy/scipy/blob/v1.5.1/scipy/integrate/_ivp/rk.py#L183-L273
        """

        t = self.t
        y = self.y

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        # changed here to hard minimal stepsize.
        min_step = np.max([self.min_step_hard, 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)])

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        step_accepted = False
        step_rejected = False

        while not step_accepted:
                        
            if h_abs < min_step:
                #h_abs = min_step # 
                #step_accepted = True # 
                return False, self.TOO_SMALL_STEP
            
            
            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            y_new, f_new = rk_step(self.fun, t, y, self.f, h, self.A,
                                   self.B, self.C, self.K)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = self._estimate_error_norm(self.K, h, scale)

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm ** self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                step_accepted = True
            else:
                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm ** self.error_exponent)
                step_rejected = True

        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True, None

######################################################################################
"""
model
"""
# Ca non-linearity
def sigmoid(x,k,x0):
    """
    sigmoid that converts calcium concentration to effective Calcium
    x0: half point 
    k: slope
    """
    y = 1/(1+np.exp(-k*(x-x0)))
    return y

class CaInterpolation():
    """
    simple linear interpolation class for ca signal
    necessary for the ODE solver, st Ca can be evaluated at any timepoint
    """
    def __init__(self,t_vector, Ca):
        self.t_vector = t_vector
        self.Ca = Ca
        self.interpolate_ca = scp.interpolate.interp1d(np.hstack([t_vector[0]-10,t_vector,t_vector[-1]+10]), np.hstack([Ca[0],Ca,Ca[-1]]))
        
    def get_ca_t(self,t):
        if type(t) == np.ndarray:
            return self.interpolate_ca(t)

        else:
            if t < self.t_vector[0]:
                return self.Ca[0]
            elif t > self.t_vector[-1]:
                return self.Ca[-1]
            else:
                return self.interpolate_ca(t)

def evaluate_ode(t, y, max_rates, max_pools, ca_interpolation):
    """
    evaluates the ode of the ribbon dynamic
    ---
    t: time (only ca_interpolation is actually depending on t)
    y: actual y(t) ODE variables (RP, IP, RRP, Exo)
    max_rates, max_pools: ribbon parameters. constant from solver perspective
    ca_interpolation: CaInterpolation class object.
    
    returns: dRP_dt, dIP_dt, dRRP_dt, dExo_dt
    """
    # unpack parameters
    RP, IP, RRP, Exo = y # actual y(t) ODE variables
    J_RP_IP_max, J_IP_RRP_max,J_RRP, F_Endo = max_rates
    RP_max,IP_max,RRP_max = max_pools 
    
    # get Calcium at tpt t
    Ca_t = ca_interpolation.get_ca_t(t)

    # evaluate changing rates
    J_RP_IP = max(0, (J_RP_IP_max * (1 - IP / IP_max) * RP / RP_max)) 
    J_IP_RRP = max(0, (J_IP_RRP_max * IP / IP_max * (1 - RRP / RRP_max))) 
    J_Exo = max(0, (J_RRP * RRP / RRP_max * Ca_t) )
    J_Exo_RP = (F_Endo * Exo)
        
    # evaluate dy/dt 
    dRP_dt =  J_Exo_RP - J_RP_IP
    dIP_dt =  J_RP_IP - J_IP_RRP
    dRRP_dt = J_IP_RRP - J_Exo
    dExo_dt =  J_Exo - J_Exo_RP
    
    return np.array([dRP_dt, dIP_dt, dRRP_dt, dExo_dt])


def solve_ribbon_ode(Ca_raw, J_RP_IP_max_pR, J_IP_RRP_max_pR, J_RRP_pR, k, x_half, IP_max_raw, RRP_max_raw,
                     dt=0.032,
                     batch_size=1, random_state=None, 
                     return_ode_res=False,
                    solver=None):
    """
    solves the ribbon ODEs with RK23, maximal stepsize of dt
    ---
    Ca_raw: signal with appropriate dt 
    ribbon_parameters: the max.-changing parameters are should be given in unit: 1/sec
    ---
    returns:  Jexo (with dt = 32ms) (optional output of ODE-solver)
    """
    # time parameters
    #dt=0.032
    adaptation_time = 4 # in sec (should be devisible by dt)
    
    # maximal ves pools as params
    RP_max = 35186 #round(Vol * D_V) # value for r=2um
    RRP_max = RRP_max_raw  # *nR
    IP_max = IP_max_raw  # *nR

    
    # scale changing rates to total ribbon number and scale appropriat to unit:1/sec
    J_RP_IP_max = J_RP_IP_max_pR# *nR
    J_IP_RRP_max = J_IP_RRP_max_pR# *nR
    F_Endo = 0.0001
    J_RRP = J_RRP_pR # *nR
    
    # NL of Ca
    Ca_RRP = sigmoid(Ca_raw, k, x_half)  
    
    t_span_pre = (-adaptation_time,0)
    t_span = (0, len(Ca_RRP)*dt)
    
    # define Ca interpolation class for pre run
    t_pre = np.arange(-adaptation_time, 0, dt)
    Ca_pre = np.ones(len(t_pre))*np.mean(Ca_RRP[:4]) # set with adaptation value
    ca_interpolation_pre =  CaInterpolation( t_pre, Ca_pre)
    
    # define Ca interpolation class for main run
    t = np.arange(0, len(Ca_RRP)*dt, dt)
    ca_interpolation =  CaInterpolation( t, Ca_RRP)
    
    # pack parameters
    max_rates = np.array([J_RP_IP_max, J_IP_RRP_max,J_RRP, F_Endo])
    max_pools = np.array([RP_max,IP_max,RRP_max])
    
    # initial value for pre run
    y0 = np.array([RP_max,IP_max,RRP_max,0]) * 0.8 # [RP, IP, RRP,Exo]
   
    # run for adaptation to get good initial values. larger max_step size possible.
    min_step_hard =  dt/10 #0.001 #1ms
    res_prerun = scp.integrate.solve_ivp(evaluate_ode, 
                                  t_span_pre, 
                                  y0, 
                                  args=(max_rates, max_pools, ca_interpolation_pre),
                                  #max_step =dt,
                                  first_step=dt,
                                  method= MyRK23,#'RK23',
                                  rtol=1e-3,#default:1e-3,
                                  atol=1e-6,#default:1e-6
                                  min_step_hard=min_step_hard
                                 )
    
    y0 = res_prerun.y[:,-1]
    # solve ODE #RP, IP, RRP, Exo = y
    res = scp.integrate.solve_ivp(evaluate_ode, 
                                  t_span, 
                                  y0, 
                                  args=(max_rates, max_pools, ca_interpolation),
                                  max_step =dt,
                                  min_step_hard=min_step_hard,
                                  #first_step=dt/2,
                                  method= MyRK23,#'RK23',
                                  rtol=1e-3,#default:1e-3,
                                  atol=1e-6,#default:1e-6
                                  #t_eval= np.arange(t_span[0],t_span[1],dt) 
                                 )
    
    # check if ode solver was successfull
    if res.success:
        # reconstruct Jexo and interpolate to initial time axis
        J_Exo_raw =  (J_RRP * res.y[2] / RRP_max * ca_interpolation.get_ca_t(res.t)) 
        J_Exo = scp.interpolate.interp1d(res.t, J_Exo_raw)(np.arange(0, len(Ca_RRP)*dt, dt))
    else:
        J_Exo = np.zeros(len(Ca_RRP))*np.nan
        
    if return_ode_res:
        return J_Exo, res
    else:
        return J_Exo



    
    
    
    