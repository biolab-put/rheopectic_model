from re import U
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from scipy.optimize import minimize
import math
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete, lti, dlti, dstep
from scipy.ndimage.interpolation import shift
from scipy.optimize import show_options
from scipy.optimize import differential_evolution
from scipy.optimize import NonlinearConstraint
from scipy.integrate import ode



delta = 0.03 #30mm - początkowe wydłużenie mięśnia

def objective(x, X0, time_vector, active_force, sim_dt, reference_force):
    km = x[0]
    kt = x[1]
    m = x[2]
    c = x[3]
    lm0 = kt * delta / (km+kt)
    Y0 = [lm0,0]
    muscle_data = ode_muscle_response(Y0, time_vector, active_force,m,km,kt,c,sim_dt)
    estimated_force = pd.DataFrame(data = {'timestamp': time_vector,'estimated force' : muscle_data['estimated force']})
    return np.sum((estimated_force['estimated force']-reference_force['reference force'])**2)

def get_damping_ratio(km,kt,m,c):
    natural_frequency = np.sqrt(np.abs(kt+km)/np.abs(m))
    damping_ratio = c /(2*m*natural_frequency)
    return damping_ratio

'''
def lt_constrain(x):
    km = x[0]
    kt = x[1]
    lt = km*delta/(kt+km)
    return lt 

def lm_constrain(x):
    km = x[0]
    kt = x[1]
    delta_lm = kt * delta / (km+kt)
    return delta_lm 

def kt_constrain(x):
    km = x[0]
    kt = x[1]
    return kt/km
'''

def damping_ratio_constrain(x):
    km = x[0]
    kt = x[1]
    m = x[2]
    c = x[3]
    damping_ratio = get_damping_ratio(km,kt,m,c)
    return damping_ratio

def parabolic_twitch(t,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt):
    single_twitch_t = np.arange(0,twitch_duration,sim_dt)
    pulse = np.zeros(int(1/(sim_dt*twitch_frequency)))
    pulse[(single_twitch_t/sim_dt).astype(np.int8)] = twitch_amplitude * np.sin(np.pi * single_twitch_t / twitch_duration)
    repeat = np.ceil((t[-1]+sim_dt)/(1/twitch_frequency))
    twitch_train = np.tile(pulse, int(repeat))
    twitch_train = twitch_train[0:len(t)]
    twitch_train = np.roll(twitch_train,int((twitch_delay)/sim_dt))
    twitch_train[0:int((twitch_delay+sim_dt)/sim_dt)] = 0
    return twitch_train

def solve_muscle_dynamics(t,x, active_force,m,km,kt,c,sim_dt):
    lm, dlm = x
    d2lm = 1/m * (-np.abs(c)*dlm-np.abs(km)*lm+np.abs(kt)*(delta - lm)-active_force[int(t/sim_dt)])
    return [dlm,d2lm]


def ode_muscle_response(X0,time_vector, active_force,m,km,kt,c,sim_dt):
    solution = solve_ivp(solve_muscle_dynamics, t_span=[0, time_vector[-1]], y0=X0, t_eval = time_vector, args=(active_force, m, km,kt,c, sim_dt), method = 'LSODA',first_step = sim_dt, max_step = sim_dt, atol = 1e90, rtol= 1e90)
    estimated_force = np.abs(kt) * (delta - solution.y[0,:])
    estimated_force = estimated_force - estimated_force[0]
    return pd.DataFrame(data = {'timestamp': time_vector, 'estimated force': estimated_force, 'lm' : solution.y[0,:], 'dlm': solution.y[1,:]})

def load_data(filename):
    data = pd.read_csv(filename)
    return data

def get_rising_time(signal,sim_dt):
    amplitude = np.max(signal)
    t90 = np.where(signal >= 0.9*amplitude)[0][0]
    t10 = np.where(signal >= 0.1*amplitude)[0][0]
    return (t90-t10)*sim_dt

def get_falling_time(signal,sim_dt):
    reversed_signal = np.flipud(signal)
    return get_rising_time(reversed_signal,sim_dt)

def c_from_damping_ratio(damping_ratio, m,kt,km):
    return  damping_ratio * 2 * m * np.sqrt((kt+km)/m)

def muscle_state():
    sim_dt = 0.001
    sim_time = 1.2
    zeros_time = 1.2 - 0.3 #sim_time - 0.5
    time_vector = np.arange(0,sim_time,sim_dt)
    twitch_duration = 0.016
    twitch_delay = 0.06
    twitch_frequency = 0.5
    twitch_amplitude = 1
    km = 0.01
    kt = 0.1
    m = 0.002 #2mg
    damping_ratio = 0.9
    damping_ratio_margin = 0.1
    c = c_from_damping_ratio(damping_ratio,m,kt,km) #+0.02
    lm0 = kt * delta / (km+kt)
    lt0 = delta - lm0

    X0 = [lm0, 0]

    print('Poczatkowe naprezenie: ', kt*(lt0), '[N].')
    print('Poczatkowe naprezenie: ', km*(lm0), '[N].')
    print('Poczatkowa dlugosc miesnia: ', lm0, '[m]')
    print('Poczatkowa dlugosc sciegna: ', lt0, '[m]')

    active_force = parabolic_twitch(time_vector,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    
    estimated_muscle_data = ode_muscle_response(X0,time_vector, active_force,m,km,kt,c,sim_dt)

    
    filename = '1006j_trial_04'
    reference_data = load_data(f'./{filename}')
    reference_force = pd.DataFrame(data = {'timestamp': time_vector, 'reference force' : reference_data['force'][0:int(sim_time/sim_dt)]})
    reference_force['reference force'] = (reference_force['reference force'] - reference_force['reference force'][0])
    reference_force['reference force'] = reference_force['reference force']/1000
    reference_force['reference force'][int(zeros_time/sim_dt)::] = 0

    print('Czas narastania sygnalu referencyjnego: ', get_rising_time(reference_force['reference force'],sim_dt), '[s]')
    print('Czas narastania sygnalu estymowanego: ', get_rising_time(estimated_muscle_data['estimated force'],sim_dt), '[s]')
    print('Czas opadania sygnalu referencyjnego: ', get_falling_time(reference_force['reference force'],sim_dt), '[s]')
    print('Czas opadania sygnalu estymowanego: ', get_falling_time(estimated_muscle_data['estimated force'],sim_dt), '[s]')
    
    plt.plot(time_vector, reference_force['reference force'])
    #plt.plot(time_vector, estimated_muscle_data['lm'])
    plt.plot(time_vector, estimated_muscle_data['estimated force'])
    #plt.plot(time_vector, active_force)
    plt.show()
    exit()

    x0 = [km,kt,m,c]

    drc = NonlinearConstraint(damping_ratio_constrain,damping_ratio - damping_ratio_margin, damping_ratio + damping_ratio_margin)
   
    result = differential_evolution(objective,x0 = x0, args = (X0, time_vector, active_force, sim_dt, reference_force),constraints=(drc), bounds = ((0.0001,0.01), (0.01,10),(0.0001,0.099),(0.001, 120)),disp= True)
    #result = minimize(objective, x0, args = (X0, time_vector, active_force, sim_dt, reference_force), method='L-BFGS-B',bounds = ((0.1, 12000), (0.1, 200000000),(0.0001,0.99),(0.0001, 100)),tol=1e-12)#, jac=derivative)
    print(result)
    fitted_muscle_data = ode_muscle_response(X0,time_vector, active_force,result.x[2],result.x[0],result.x[1],result.x[3],sim_dt)
    ax = fitted_muscle_data.plot(x="timestamp", y="estimated force")
    reference_force.plot(x='timestamp', y="reference force",ax=ax)
    plt.show()
    


muscle_state()
