from hill_muscle_model import hill_muscle_model
from hill_muscle_model import rheopectic_hill_muscle_model
from muscle_active_force import muscle_active_force
import numpy as np
from matplotlib import pyplot as plt 
from scipy.optimize import NonlinearConstraint
from scipy.optimize import differential_evolution
import pandas as pd

def objective(x,muscle_model,time_vector, active_force,reference_force):
    muscle_model.set_parameters(x)
    lm0 = muscle_model.get_initial_length()
    X0 = [lm0,0]
    error = 0
    for i, (time,input,reference) in enumerate(zip(time_vector,active_force,reference_force)):
        muscle_data = muscle_model.muscle_response(X0,time,input)
        error += 1/len(muscle_data) * np.sum((muscle_data-reference)**2)
        if(i==0):
            error *= 10
    return error

def load_data(filename):
    data = pd.read_csv(filename)
    return data

def prepare_reference_data(filenames,sim_dt):
    reference_0_steady_time = 0.09
    reference_1_steady_time = 0.6
    zeros_time = 0.3 #percent of the steady time

    reference_data = [load_data(f'./{filename}') for filename in filenames]

    for reference in reference_data:
        reference['force'] = reference['force'] - reference['force'][0] # delete bias
        reference['force'] = reference['force'] / 1000 # convert from mN to N
        print(len(reference['force']))

    reference_data[0]['force'][int(reference_0_steady_time/sim_dt)::] = 0
    reference_data[1]['force'][int(reference_1_steady_time/sim_dt)::] = 0
    reference_data[0] = reference_data[0][0:int((reference_0_steady_time+(zeros_time*reference_0_steady_time))/sim_dt)]
    reference_data[1] = reference_data[1][0:int((reference_1_steady_time+(zeros_time*reference_1_steady_time))/sim_dt)]
    reference_force = [reference_data[0]['force'],reference_data[1]['force']]
    return reference_force

def muscle_identification(muscle_model,time_vector,active_force,reference_force,damping_ratio,damping_ration_margin,bounds,threads,disp_debug):
    x0 = muscle_model.get_parameters()
    drc = NonlinearConstraint(hill_muscle_model.get_damping_ratio, damping_ratio - damping_ration_margin, damping_ratio + damping_ration_margin)
    src = NonlinearConstraint(hill_muscle_model.get_stiffness_ratio, 10, 1000)
    result = differential_evolution(objective,x0 = x0,args = (muscle_model, time_vector, active_force, reference_force),constraints=(drc,src),bounds=bounds,workers = threads, disp = disp_debug)
    return result

def muscle_optimization():
    # muscle parameters
    sim_dt = 0.0001
    twitch_duration = 0.015
    twitch_delay = 0.005
    twitch_frequency = 40
    twitch_amplitude = 0.1
    km = 0.1
    kt = 1
    m = 0.002
    c = 1 * 2 * m * np.sqrt((kt+km)/m)
    delta = 0.03
    F_K = 1
    F_delta = 1
    # identification parameters
    disp_debug = True
    km_bound = (0.0001,1000)
    kt_bound = (0.0001,1000)
    m_bound = (0.0001,0.0799)
    c_bound = (0.001,120)
    FK_bound = (0.1,20)
    Fdelta_bound = (0.01,10)
    #bounds = (km_bound,kt_bound,m_bound,c_bound)
    bounds = (km_bound,kt_bound,m_bound,c_bound,FK_bound,Fdelta_bound)
    threads = 3
    damping_ratio = 1.
    damping_ratio_margin = 0.1

    # create hill model
    muscle_model = hill_muscle_model(km,kt,m,c,F_K,F_delta,delta,sim_dt)
    #muscle_model = hill_muscle_model(km,kt,m,c,delta,sim_dt)
    lm0 = muscle_model.get_initial_length()
    X0 = [lm0,0]
    # load reference data
    filenames = ['1006j_trial_04','1006j_trial_05']
    # take both signals separately during optimization
    reference_forces = prepare_reference_data(filenames,sim_dt)
    time_vector_0 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    input_0 = muscle_active_force.parabolic_twitch(time_vector_0,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    input_1 = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    input_1[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal
    time_vectors = [time_vector_0,time_vector_1]

    active_forces = [input_0,input_1]
    result = muscle_identification(muscle_model,time_vectors,active_forces,reference_forces,damping_ratio,damping_ratio_margin,bounds,threads,disp_debug)
    muscle_model.set_parameters(result.x)
    lm0 = muscle_model.get_initial_length()
    X0 = [lm0,0]
    print(result)
    fitted_muscle_data_0 = muscle_model.muscle_response(X0,time_vectors[0],active_forces[0])
    fitted_muscle_data_1 = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1])

    plt.figure(0)
    plt.plot(time_vectors[0], fitted_muscle_data_0)
    plt.plot(time_vectors[0], reference_forces[0])
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.figure(1)
    plt.plot(time_vectors[1], fitted_muscle_data_1)
    plt.plot(time_vectors[1], reference_forces[1])
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.show()

def muscle_simulation():
    sim_dt = 0.0001
    sim_time = 0.8
    time_vector = np.arange(0,sim_time-sim_dt,sim_dt)
    twitch_duration = 0.005
    twitch_delay = 0.005
    twitch_frequency = 40
    twitch_amplitude = 0.5
    km = 1.08057790e+01
    #km = 0.1
    kt = 2.38328345e+02
    #kt = 1
    m = 7.56145294e-02
    #m = 0.002
    c = 8.76722115e+00
    #c = 1 * 2 * m * np.sqrt((kt+km)/m)
    delta = 0.03
    F_K = 90
    F_delta = 1
    active_force = muscle_active_force.parabolic_twitch(time_vector,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    active_force[int(0.525/sim_dt)::]=0
    #muscle_model = hill_muscle_model(km,kt,m,c,F_K,F_delta,delta,sim_dt)
    muscle_model = hill_muscle_model(km,kt,m,c,delta,sim_dt)
    lm0 = muscle_model.get_initial_length()
    X0 = [lm0,0]
    muscle_force = muscle_model.muscle_response(X0,time_vector,active_force)
    plt.plot(time_vector,muscle_force)
    plt.show()

def rheopectic_simulation():
    sim_dt = 0.0001
    sim_time = 0.8
    time_vector = np.arange(0,sim_time-sim_dt,sim_dt)
    twitch_duration = 0.005
    twitch_delay = 0.005
    twitch_frequency = 40
    twitch_amplitude = 1
    km = 0.1
    kt = 1
    m = 0.002
    delta = 0.03
    A = 1
    B = 1
    C = 2
    D = 1
    max_c = 0.05
    min_c = 0.001
    c0 = 0
    k1 = 1
    k2 = 5
    active_force = muscle_active_force.parabolic_twitch(time_vector,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    active_force[int(0.525/sim_dt)::]=0
    muscle_model = rheopectic_hill_muscle_model(km,kt,m,c0,min_c,max_c,k1,k2,A,B,C,D,delta,sim_dt)
    lm0 = muscle_model.get_initial_length()
    X0 = [lm0,0,0]
    muscle_force = muscle_model.muscle_response(X0,time_vector,active_force)
    plt.plot(time_vector,muscle_force)
    plt.show()


if __name__=="__main__":
    #muscle_simulation()
    muscle_optimization()
    #rheopectic_simulation()
