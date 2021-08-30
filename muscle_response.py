import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from scipy.optimize import minimize


def objective(x, X0, time_vector, active_force, sim_dt, reference_force):
    km = x[0]
    kt = x[1]
    m = x[2]
    c = x[3]
    muscle_data = muscle_response(X0, time_vector, active_force,m,km,kt,c,sim_dt)
    estimated_force = pd.DataFrame(data = {'timestamp': time_vector,'estimated force' : muscle_data['estimated force']})
    return np.sum((estimated_force['estimated force']-reference_force['reference force'])**2)

def muscle_response(X0,time_vector, active_force,m,km,kt,c,sim_dt):
    a = np.array([[0, 1], [(-km/m-kt/m), -c/m]])
    b = np.array([[0], [1/m]])
    #change from lm to force and convert to mN
    c = np.array([[kt, 0]])
    d = np.array([[0]])
    sys = signal.StateSpace(a, b, c, d)
    #active_force = twitch_amplitude * signal.unit_impulse(len(time_vector), (np.arange(twitch_delay,twitch_delay+twitch_duration,sim_dt)/sim_dt).astype(int))
    tout, yout, xout = signal.lsim(system = sys,U = active_force, T = time_vector,X0 = X0)
    return pd.DataFrame(data = {'timestamp': tout, 'estimated force': yout, 'lm' : xout[:,0], 'dlm': xout[:,1]})

def load_data(filename):
    data = pd.read_csv(filename)
    return data

def muscle_state():
    sim_dt = 0.001
    sim_time = 1
    zeros_time = 0.6 #sim_time - 0.5
    time_vector = np.arange(0,sim_time,sim_dt)
    twitch_duration = 0.025
    twitch_delay = 0.06
    twitch_frequency = 0
    twitch_amplitude = 10
    km = 0.1
    kt = 1
    c = 0.03
    m = 0.000650
    X0 = [0, 0]


    active_force = twitch_amplitude * signal.unit_impulse(len(time_vector), (np.arange(twitch_delay+sim_dt,twitch_delay+twitch_duration,sim_dt)/sim_dt).astype(int))
    estimated_muscle_data = muscle_response(X0,time_vector, active_force,m,km,kt,c,sim_dt)
    filename = '1006j_trial_0'
    reference_data = load_data(f'../{filename}')
    #reference_data = reference_data[reference_data['timestamp'] <= sim_time]
    #reference_data['force'][int(zeros_time/sim_dt)::] = 0
    reference_force = pd.DataFrame(data = {'timestamp': time_vector, 'reference force' : reference_data['force'][0:int(sim_time/sim_dt)]})
    reference_force['reference force'] = (reference_force['reference force'] - reference_force['reference force'][0])/1000
    reference_force['reference force'][int(zeros_time/sim_dt)::] = 0
    
    #plt.plot(time_vector, active_force)
    #plt.show()

    reference_force.plot(x='timestamp', y="reference force")
    plt.show()

    #estimated_muscle_data.plot(x="timestamp", y="estimated force")
    #plt.show()

    x0 = [km,kt,m,c]
    result = minimize(objective, x0, args = (X0, time_vector, active_force, sim_dt, reference_force), method='SLSQP',bounds = ((0, None), (0, None),(0, None),(0, None)))#, jac=derivative)
    fitted_muscle_data = muscle_response(X0,time_vector, active_force,result.x[2],result.x[0],result.x[1],result.x[3],sim_dt)

    ax = fitted_muscle_data.plot(x="timestamp", y="estimated force")
    reference_force.plot(x='timestamp', y="reference force",ax=ax)
    print(result)
    plt.show()
    


muscle_state()
