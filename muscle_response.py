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

def objective(x, X0, time_vector, active_force, sim_dt, reference_force):
    km = x[0]
    kt = x[1]
    m = x[2]
    c = x[3]
    muscle_data = muscle_response(X0, time_vector, active_force,m,km,kt,c,sim_dt)
    estimated_force = pd.DataFrame(data = {'timestamp': time_vector,'estimated force' : muscle_data['estimated force']})
    return np.sum((estimated_force['estimated force']-reference_force['reference force'])**2)

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
    #print(pulse)
    #pulse[int((twitch_delay+twitch_duration)/sim_dt)::] = 0
    #return pulse
    #train_twitch_t = twitch_delay:1/twitch_frequency:t(end)
    #np.arange(twitch_delay,1/twitch_frequency,sim_dt)

def muscle_response(X0,time_vector, active_force,m,km,kt,c,sim_dt):
    a = np.array([[0, 1], [(-abs(km/m)-abs(kt/m)), -abs(c/m)]])
    b = np.array([[0], [abs(1/m)]])
    #change from lm to force and convert to mN
    c = np.array([[abs(kt)*1000, 0]])
    d = np.array([[0]])
    sys = signal.StateSpace(a, b, c, d)
    sys = sys.to_discrete(sim_dt)
    #tout, yout, xout = signal.lsim(system = sys,U = active_force, T = time_vector,X0 = X0)
    tout, yout, xout = signal.dlsim(system = sys,u = active_force, t = time_vector,x0 = X0)
    return pd.DataFrame(data = {'timestamp': tout, 'estimated force': yout[:,0], 'lm' : xout[:,0], 'dlm': xout[:,1]})

def load_data(filename):
    data = pd.read_csv(filename)
    return data

def muscle_state():
    sim_dt = 0.001
    sim_time = 5.1
    zeros_time = 1.2 - 0.5 #sim_time - 0.5
    time_vector = np.arange(0,sim_time,sim_dt)
    twitch_duration = 0.025
    twitch_delay = 0
    twitch_frequency = 4
    twitch_amplitude = 10
    km = 0.1848                   
    kt = 0.0087
    m = 0.0038
    c =  0.0350  
    X0 = [0, 0]


    #active_force = twitch_amplitude * signal.unit_impulse(len(time_vector), (np.arange(twitch_delay+sim_dt,twitch_delay+twitch_duration,sim_dt)/sim_dt).astype(int))
 

    filenames = ['1006j_trial_0', '1006j_trial_5', '1006j_trial_0']
    
    

    reference_data = [load_data(f'./{filename}') for filename in filenames]
    zeros = pd.DataFrame(np.zeros((2000,reference_data[0].shape[1])),columns=reference_data[0].columns)
    reference_data[0]['force'] = reference_data[0]['force'] - reference_data[0]['force'][0]
    reference_data[2]['force'] = reference_data[2]['force'] - reference_data[2]['force'][0]
    reference_data[0] = reference_data[0][0:600]
    reference_data[1] = reference_data[1][0:6000]
    reference_data[2] = reference_data[2][0:600]

    time_vector_twitch_start = np.arange(0,len(reference_data[0]['force']) *sim_dt,sim_dt)
    active_force_twitch_start = parabolic_twitch(time_vector_twitch_start,twitch_duration,0.06,twitch_amplitude, 1, sim_dt)

    time_vector_twitches_middle = np.arange(0,len(reference_data[1]['force']) *sim_dt,sim_dt)
    active_force_twitches_middle = parabolic_twitch(time_vector_twitches_middle,twitch_duration,0.06,twitch_amplitude, twitch_frequency, sim_dt)
    # HARDCORE FIX
    active_force_twitches_middle[5200::] = 0

    time_vector_twitch_stop = np.arange(0,len(reference_data[2]['force']) *sim_dt,sim_dt)
    active_force_twitch_stop = parabolic_twitch(time_vector_twitch_stop,twitch_duration,0.06,twitch_amplitude, 1, sim_dt)

    active_force = np.concatenate((active_force_twitch_start,zeros['force'].values, active_force_twitches_middle,zeros['force'].values, active_force_twitch_stop, zeros['force'].values))
    reference_data = pd.concat([reference_data[0],zeros,reference_data[1],zeros,reference_data[2],zeros], ignore_index=True)
    
    time_vector = np.arange(0,len(reference_data['force'])*sim_dt,sim_dt)

    plt.plot(time_vector, active_force)
    plt.plot(time_vector, reference_data['stimDig'])

    #plt.show()
    #reference_data = reference_data[reference_data['timestamp'] <= sim_time]
    #reference_data['force'][int(zeros_time/sim_dt)::] = 0
    #reference_force = pd.DataFrame(data = {'timestamp': time_vector, 'reference force' : reference_data['force'][0:int(sim_time/sim_dt)]})
    #reference_force['reference force'] = (refere
    # reference_force['reference force'] - reference_force['reference force'][0])
    #reference_force['reference force'][int(zeros_time/sim_dt)::] = 0
    
    #plt.plot(time_vector, active_force)
    #plt.show()

    #reference_force.plot(x='timestamp', y="reference force")

    plt.plot(time_vector, reference_data['force'])
    #plt.show()

    estimated_muscle_data = muscle_response(X0,time_vector, active_force,m,km,kt,c,sim_dt)
    #estimated_muscle_data.plot(x="timestamp", y="estimated force")
    #plt.show()
    x0 = [km,kt,m,c]
    #reference_force = reference_data['force']
    reference_force = pd.DataFrame(data = {'timestamp': time_vector, 'reference force' : reference_data['force']})
    result = minimize(objective, x0, args = (X0, time_vector, active_force, sim_dt, reference_force), method='L-BFGS-B',bounds = ((0, None), (0, None),(0, None),(0, None)))#, jac=derivative)
    fitted_muscle_data = muscle_response(X0,time_vector, active_force,result.x[2],result.x[0],result.x[1],result.x[3],sim_dt)
    

    #ax = fitted_muscle_data.plot(x="timestamp", y="estimated force")
    plt.plot(time_vector, fitted_muscle_data['estimated force'])
    plt.legend(['Active force', 'Distribution of stimuli', 'Reference muscle force', 'Estimated muscle force'],loc ='upper right')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    #reference_force.plot(x='timestamp', y="reference force",ax=ax)
    print(result)
    plt.show()
    


muscle_state()
