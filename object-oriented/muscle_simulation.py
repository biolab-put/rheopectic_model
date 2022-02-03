from re import T
from time import time
from turtle import color
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
    X0 = muscle_model.get_X0()
    error = 0
    for i, (time,input,reference) in enumerate(zip(time_vector,active_force,reference_force)):
        muscle_data,_ = muscle_model.muscle_response(X0,time,input)
        #error += 1/len(muscle_data) * np.sum((muscle_data-reference)**2)

        #error += np.sum((muscle_data-reference)**2)
        if(i==1):
            #print(muscle_data[0:0.3/0.0001])
            error += np.sum((muscle_data[0:int(0.45/0.0001)] - reference[0:int(0.45/0.0001)])**2)
            error *= 1
            error += np.sum((muscle_data[int(0.45/0.0001):int(0.53/0.0001)] - reference[int(0.45/0.0001):int(0.53/0.0001)])**2)
            error *= 1000
            error += np.sum((muscle_data[int(0.53/0.0001)::] - reference[int(0.53/0.0001)::])**2)
            error *= 1
            error *= 1/len(muscle_data)
        #error += 1/len(muscle_data) * np.sum((muscle_data-reference)**2)
        #error += np.sum((muscle_data-reference)**2)
        if(i==0):                   
            error += np.sum((muscle_data-reference)**2)
            error *= 1/len(muscle_data)
            error *= 1
    return error

def objective_sigmoid(x,X0,muscle_model,time_vector, active_force,reference_force):
    F_K = x[0]
    F_delta = x[1]
    error = 0
    for i, (time,input,reference) in enumerate(zip(time_vector,active_force,reference_force)):
        muscle_data = muscle_model.muscle_response(X0,time,input,F_K,F_delta)
        #error += 1/len(muscle_data) * np.sum((muscle_data-reference)**2)
        #error += np.sum((np.log((muscle_data-reference)+1))**2)
        error += np.sum((muscle_data-reference)**2)
        if(i==0):
            error *= 0
    return error

def load_data(filename):
    data = pd.read_csv(filename)
    return data

def prepare_reference_data_in(filename,sim_dt,steady_time = 0, zeros_time = 0):
    reference_data = load_data(filename)
    reference_data['force'] = reference_data['force'] - reference_data['force'][0] # delete bias
    reference_data['force'] = reference_data['force'] / 1000 # convert from mN to N
    if(steady_time > 0):
        #reference_data['force'][int(steady_time/sim_dt)::] = 0
        reference_data.loc[int(steady_time/sim_dt)::, ['force']] = 0
        reference_data = reference_data[0:int((steady_time+(zeros_time*steady_time))/sim_dt)]
    return reference_data

def prepare_reference_data(filenames,sim_dt):
    reference_0_steady_time = 0.09
    reference_1_steady_time = 0.6
    zeros_time = 0.3 #percent of the steady time
    
    reference_data_0 = prepare_reference_data_in(filenames[0],sim_dt,reference_0_steady_time,zeros_time)
    reference_data_1 = prepare_reference_data_in(filenames[1],sim_dt,reference_1_steady_time,zeros_time)
    reference_force = [reference_data_0['force'],reference_data_1['force']]
    '''
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
    '''
    return reference_force

def muscle_identification(muscle_model,time_vector,active_force,reference_force,damping_ratio,damping_ration_margin,bounds,threads,disp_debug):
    x0 = muscle_model.get_parameters()
    drc = NonlinearConstraint(hill_muscle_model.get_damping_ratio, damping_ratio - damping_ration_margin, damping_ratio + damping_ration_margin)
    src = NonlinearConstraint(hill_muscle_model.get_stiffness_ratio, 10, 1000)
    result = differential_evolution(objective,x0 = x0,args = (muscle_model, time_vector, active_force, reference_force),constraints=(drc,src),bounds=bounds,workers = threads, disp = disp_debug,polish=False)
    return result

def muscle_optimization():
    # muscle parameters
    sim_dt = 0.0001
    twitch_duration = 0.01
    twitch_delay = 0.0025
    twitch_frequency = 40
    twitch_amplitude = 1
    km = 0.1
    kt = 1
    m = 0.002
    c = 1 * 2 * m * np.sqrt((kt+km)/m)
    delta = 0.01
    F_K = 1
    F_delta = 1
    # identification parameters
    disp_debug = True
    km_bound = (0.0001,1000)
    kt_bound = (0.0001,1000)
    m_bound = (0.0001,0.0799)
    c_bound = (0.001,120)
    FK_bound = (0.01,100)
    Fdelta_bound = (0.001,10)
    #bounds = (km_bound,kt_bound,m_bound,c_bound)
    bounds = (km_bound,kt_bound,m_bound,c_bound)
    threads = 12
    damping_ratio = 1.
    damping_ratio_margin = 0.1

    # create hill model
    muscle_model = hill_muscle_model(km,kt,m,c,delta,sim_dt)
    #muscle_model = hill_muscle_model(km,kt,m,c,delta,sim_dt)
    X0 = muscle_model.get_X0()
    # load reference data
    filenames = ['947kLHLf/947kLHLf_trial_04','947kLHLf/947kLHLf_trial_07']
    # take both signals separately during optimization
    #reference_forces = prepare_reference_data(filenames,sim_dt)
    reference_force_0 = prepare_reference_data_in(filenames[0],sim_dt,steady_time=1.5,zeros_time=0.3)
    reference_force_1 = prepare_reference_data_in(filenames[1],sim_dt,steady_time=0.6,zeros_time=0.3)
    reference_forces = [reference_force_0['force'],reference_force_1['force']]
    time_vector_0 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    input_0 = muscle_active_force.parabolic_twitch(time_vector_0,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    input_1 = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    input_1[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal
    time_vectors = [time_vector_0,time_vector_1]
    active_forces = [input_0,input_1]



    #plt.figure(1)
    #plt.plot(time_vector_1,muscle_data,'tab:blue')
    #plt.plot(time_vector_1,reference_force_1['force'], 'tab:orange')

    #plt.xlabel('Time [s]')
    #plt.ylabel('Force [N]')
    #plt.show()
    #exit()
    result = muscle_identification(muscle_model,time_vectors,active_forces,reference_forces,damping_ratio,damping_ratio_margin,bounds,threads,disp_debug)
    muscle_model.set_parameters(result.x)
    X0 = muscle_model.get_X0()
    print(result)
    fitted_muscle_data_0,_ = muscle_model.muscle_response(X0,time_vectors[0],active_forces[0])
    fitted_muscle_data_1,_ = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1])

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

    reference_stim_dig = np.copy(reference_force_1['stimDig'])
    reference_stim_dig[reference_stim_dig == 0] = np.nan
    plt.scatter(time_vector_1,reference_stim_dig * 0,2,'black',alpha = 1)
    plt.grid(color='grey', linestyle='-', linewidth=0.3)
    plt.show()


def rheopectic_muscle_optimization():
    sim_dt = 0.0001
    twitch_duration = 0.010
    twitch_delay = 0.0025
    #input 1a
    twitch_frequency = 40
    #input 1
    low_frequency = 35
    high_frequency = 75
    #...
    twitch_amplitude = 1
    #km = 3.10865655e+00
    #kt = 4.06278815e+01
    #m = 2.08453333e-02

    km = 2.56739191e+00
    kt = 6.36386231e+01
    m = 3.07830367e-02
    c = 2.98553129e+00

    #c = 1 * 2 * m * np.sqrt((kt+km)/m)
    delta = 0.01
    #F_K = 1
    #F_delta = 1
    c0 = 0
    min_c = 0.3#????
    max_c = 120 # ???
    k1 = 30
    k2 = 30
    A = 1
    B = 1
    C = 4
    D = 2
    lambda0 = 0.4
    # identification parameters
    disp_debug = True
    max_c_bound = (10,150)
    min_c_bound = (0.1,0.3)
    k1_bound = (1,1000)
    k2_bound = (1,1000)
    C_bound = (2,5)
    D_bound = (2,5)
    c0_bound = (0,0.001)
    lambda0_bound = (0.3,0.9)
    #([  0.28579205, 109.77117007,   1.11604163,  19.96163719,
    #     1.44207403,   0.21671702])
    bounds = (min_c_bound,k1_bound,k2_bound, C_bound,D_bound,c0_bound,lambda0_bound,max_c_bound)
    threads = 12
    damping_ratio = 1.
    damping_ratio_margin = 0.1

    # create hill model
    muscle_model = rheopectic_hill_muscle_model(km,kt,m,c0,min_c,max_c,k1,k2,A,B,C,D,lambda0,delta,sim_dt)
    X0 = muscle_model.get_X0()
    # load reference data
    filenames = ['947kLHLf/947kLHLf_trial_07','947kLHLf/947kLHLf_trial_17']
    reference_force_0 = prepare_reference_data_in(filenames[0],sim_dt,steady_time=1.5,zeros_time=0.3)
    reference_force_1 = prepare_reference_data_in(filenames[1],sim_dt,steady_time=1.5,zeros_time=0.3)
    reference_forces = [reference_force_0['force'],reference_force_1['force']]
    time_vector_0 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    # prepare input data
    low_freq_duration = int((19 * ((1/low_frequency - sim_dt/1.5)))/sim_dt)
    
    high_freq_duration = int((23 * ((1/high_frequency + sim_dt/1.5)))/sim_dt)
    input_1_low_freq = muscle_active_force.parabolic_twitch(time_vector_1[0:low_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, low_frequency, sim_dt)
    input_1_high_freq = muscle_active_force.parabolic_twitch(time_vector_1[low_freq_duration:low_freq_duration+high_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, high_frequency, sim_dt)
    input_1_low_freq_end = muscle_active_force.parabolic_twitch(time_vector_1[low_freq_duration+high_freq_duration:2*low_freq_duration+high_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, low_frequency, sim_dt)

    input_1a = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    input_1a[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal
    zeros = np.zeros([len(time_vector_1)-(2*low_freq_duration+high_freq_duration)])
    input_1 = np.concatenate((input_1_low_freq, input_1_high_freq,input_1_low_freq_end,zeros), axis=0)
    X0 = muscle_model.get_X0()


    #reference_force_0 = prepare_reference_data_in(filenames[0],sim_dt,steady_time=1.5,zeros_time=0.3)
    #reference_force_1 = prepare_reference_data_in(filenames[1],sim_dt,steady_time=0.6,zeros_time=0.3)
    #reference_forces = [reference_force_0['force'],reference_force_1['force']]
    #time_vector_0 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    #input_0 = muscle_active_force.parabolic_twitch(time_vector_0,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    #time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    #input_1 = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    #input_1[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal
    time_vectors = [time_vector_0,time_vector_1]
    active_forces = [input_1a,input_1]

    # take both signals separately during optimization
    #reference_forces = prepare_reference_data(filenames,sim_dt)
    #time_vector_0 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    #input_0 = muscle_active_force.parabolic_twitch(time_vector_0,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    #time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    #input_1 = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    #input_1[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal
    #time_vectors = [time_vector_0,time_vector_1]

    #active_forces = [input_0,input_1]
    result = muscle_identification(muscle_model,time_vectors,active_forces,reference_forces,damping_ratio,damping_ratio_margin,bounds,threads,disp_debug)
    muscle_model.set_parameters(result.x)
    X0 = muscle_model.get_X0()
    print(result)
    fitted_muscle_data_0,_ = muscle_model.muscle_response(X0,time_vectors[0],active_forces[0])
    fitted_muscle_data_1,_ = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1])

    plt.figure(0)
    plt.plot(time_vectors[0], fitted_muscle_data_0)
    plt.plot(time_vectors[0], reference_forces[0])
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.figure(1)
    plt.plot(time_vectors[1], fitted_muscle_data_1)
    plt.plot(time_vectors[1], reference_forces[1])
    reference_stim_dig = np.copy(reference_force_1['stimDig'])
    reference_stim_dig[reference_stim_dig == 0] = np.nan
    plt.scatter(time_vector_1,reference_stim_dig * 0,2,'black',alpha = 1)
    plt.grid(color='grey', linestyle='-', linewidth=0.3)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.show()

def sigmoid_identification(x0,X0,muscle_model,time_vector,active_force,reference_force,bounds,threads,disp_debug):
    result = differential_evolution(objective_sigmoid,x0 = x0,args = (X0,muscle_model, time_vector, active_force, reference_force),bounds=bounds,workers = threads, disp = disp_debug)
    return result

def sigmoid_optimization():
    sim_dt = 0.0001
    twitch_duration = 0.015
    twitch_delay = 0.005
    twitch_frequency = 40
    twitch_amplitude = 0.5
    km = 1.08057790e+01
    kt = 2.38328345e+02
    m = 7.56145294e-02
    c = 8.76722115e+00
    delta = 0.01
    F_K = 10
    F_delta = 1
     # identification parameters
    disp_debug = True
    FK_bound = (0.01,20)
    Fdelta_bound = (0.01,20)
    x0 = [F_K,F_delta]
    bounds = (FK_bound,Fdelta_bound)
    threads = 6
    muscle_model = hill_muscle_model(km,kt,m,c,delta,sim_dt)
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
    result = sigmoid_identification(x0,X0,muscle_model,time_vectors,active_forces,reference_forces,bounds,threads,disp_debug)
    fitted_muscle_data_0 = muscle_model.muscle_response(X0,time_vectors[0],active_forces[0],result.x[0],result.x[1])
    fitted_muscle_data_1 = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1],result.x[0],result.x[1])

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
    time_vector = np.arange(0,sim_time,sim_dt)
    twitch_duration = 0.01
    twitch_delay = 0.0025
    twitch_frequency = 40
    twitch_amplitude = 1
    km = 1.30927678e+00
    #km = 0.1
    kt = 4.15488990e+01
    #kt = 1
    m = 2.49867981e-02
    #m = 0.002
    c = 2.27624249e+00
    #c = 1 * 2 * m * np.sqrt((kt+km)/m)
    delta = 0.01
    F_K = 20
    F_delta = 9
    active_force = muscle_active_force.parabolic_twitch(time_vector,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    active_force[int(0.525/sim_dt)::]=0
    muscle_model = hill_muscle_model(km,kt,m,c,delta,sim_dt)
    #muscle_model = hill_muscle_model(km,kt,m,c,delta,sim_dt)
    X0 = muscle_model.get_X0()
    muscle_force,[lm,dlm_dt,Fact] = muscle_model.muscle_response(X0,time_vector,active_force)
    #exit()
    plt.figure(1)
    plt.plot(time_vector,muscle_force)
    stim_dig = np.copy(active_force)
    stim_dig[stim_dig == 0] = np.nan
    plt.scatter(time_vector,stim_dig * 0,2,'black',alpha = 1)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.grid(color='grey', linestyle='-', linewidth=0.3)
    plt.figure(2)
    plt.plot(time_vector,dlm_dt)
    print('Median dlm_dt', np.median(dlm_dt))
    y_bottom,y_top= plt.gca().get_ylim()
    plt.scatter(time_vector,stim_dig * 0 - np.abs(y_bottom),2,'black',alpha = 1)
    plt.xlabel('Time [s]')
    plt.ylabel('dlm_dt')
    plt.grid(color='grey', linestyle='-', linewidth=0.3)
    plt.show()

def rheopectic_simulation():
    sim_dt = 0.0001
    sim_time = 1.6
    time_vector = np.arange(0,sim_time,sim_dt)
    twitch_duration = 0.01
    twitch_delay = 0.0025
    twitch_amplitude = 1
    twitch_frequency = 40
    low_frequency = 35
    high_frequency = 75

    km = 2.56739191e+00
    kt = 6.36386231e+01
    m = 3.07830367e-02
    c = 2.98553129e+00
    delta = 0.01
    A = 1
    B = 1
    #min_c = 1.03821004e+02
    #max_c = 3.76895335e-01
    max_viscosity = 3.41626489e+01
    min_viscosity = 2.43509362e-01
    k1 = 2.47086555e+00
    k2 = 1.48056843e+00
    C = 4.24828544e+00
    D = 2.33634510e+00
    c0 = 5.92810902e-04
    lambda0 = 6.95905615e-01
    
    #min_c_bound,k1_bound,C_bound,D_bound,c0_bound,lambda0_bound
    # 4.88715531e-01, 2.43961686e+02, 2.00293405e+00, 3.99843875e+01, 4.26249386e-02, 3.23829975e-01
    
    #active_force = muscle_active_force.parabolic_twitch(time_vector,twitch_duration,twitch_delay,twitch_amplitude, 100, sim_dt)
    #active_force[int(0.525/sim_dt)::]=0
    muscle_model = rheopectic_hill_muscle_model(km,kt,m,c0,min_viscosity,max_viscosity,k1,k2,A,B,C,D,lambda0,delta,sim_dt)
    #muscle_model = hill_muscle_model(km,kt,m,c,delta,sim_dt)

    
    # prepare input data
    low_freq_duration = int((19 * ((1/low_frequency - sim_dt/1.5)))/sim_dt)
    
    high_freq_duration = int((23 * ((1/high_frequency + sim_dt/1.5)))/sim_dt)
    input_1_low_freq = muscle_active_force.parabolic_twitch(time_vector[0:low_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, low_frequency, sim_dt)
    input_1_high_freq = muscle_active_force.parabolic_twitch(time_vector[low_freq_duration:low_freq_duration+high_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, high_frequency, sim_dt)
    input_1_low_freq_end = muscle_active_force.parabolic_twitch(time_vector[low_freq_duration+high_freq_duration:2*low_freq_duration+high_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, low_frequency, sim_dt)
    zeros = np.zeros([len(time_vector)-(2*low_freq_duration+high_freq_duration)])
    active_force = np.concatenate((input_1_low_freq, input_1_high_freq,input_1_low_freq_end,zeros), axis=0)

    X0 = muscle_model.get_X0()
    muscle_force,[lm,dlm_dt,Lambda] = muscle_model.muscle_response(X0,time_vector,active_force)
    viscosity = muscle_model.Lambda2viscosity(Lambda)
    #K = 1 - np.sqrt(max_c/min_c)
    #viscosity = max_c/((1 - K * Lambda)**2) + c0
    plt.ylabel('Force [N]')
    plt.xlabel('Time [s]')

    #print(np.diff(dlm_dt) > 0)
    #exit()
    #plt.plot(time_vector,muscle_force)
    
    #plt.scatter(dlm_dt[np.diff(dlm_dt) > 0],muscle_force[np.diff(dlm_dt) > 0],color = 'blue',s = 2)
    
    
    
    #print(np.array([1,2,3,4])[np.diff(np.array([1,2,3,4,5]),1)>0])
    #plt.scatter(dlm_dt[np.append(np.diff(dlm_dt) < 0,True)],muscle_force[np.append(np.diff(dlm_dt) < 0,True)],color = 'blue',s = 2)
    #plt.scatter(dlm_dt[np.append(np.diff(dlm_dt) > 0,True)],muscle_force[np.append(np.diff(dlm_dt) > 0,True)], color = 'orange',s = 2)
    #plt.scatter(dlm_dt,muscle_force, color = 'orange',s = 2)
    #plt.figure()
    
    test = np.convolve(dlm_dt, np.ones(500), 'valid') / 500
    #print(np.shape(time_vector[0:-99]))
    #plt.scatter(time_vector[0:-99],-viscosity[0:-99]*test,s = 2)

    plt.scatter(time_vector[0::1],-viscosity[0::1]*dlm_dt[0::1],s = 2)

    #plt.scatter(time_vector[np.append(np.diff(dlm_dt) > 0,True)],dlm_dt[np.append(np.diff(dlm_dt) > 0,True)],color = 'orange',s = 2)
    #plt.figure()
    #plt.scatter(time_vector,-c*dlm_dt,s = 2)
    #plt.scatter(time_vector[0:-99],muscle_force[0:-99],s = 2)
    plt.figure()

    #plt.scatter(test,muscle_force[0:-99],s=2)
    tc = time_vector[0:-499]
    ms = muscle_force[0:-499]
    vc = viscosity[0:-499]
    ld = Lambda[0:-499]
    #plt.scatter(tc[np.append(np.diff(test) > 0.0001,True)],test[np.append(np.diff(test) > 0.0001,True)],s = 2,color = 'orange')
    #plt.scatter(tc[np.append(np.diff(test) < -0.0001,True)],test[np.append(np.diff(test) < -0.0001,True)],s = 2, color = 'blue')

    plt.scatter(test[np.append(np.diff(test) > 0.0001,True)],ms[np.append(np.diff(test) > 0.0001,True)],s = 2,color = 'orange')
    plt.scatter(test[np.append(np.diff(test) < -0.0001,True)],ms[np.append(np.diff(test) < -0.0001,True)],s = 2, color = 'blue')

    plt.figure()
    plt.scatter(dlm_dt,Lambda,s = 2,color = 'orange')
    fig,ax = plt.subplots(2,1,tight_layout = True)
    #plt.scatter(time_vector,dlm_dt)
    ax[0].scatter(tc,test,s = 2)
    ax[0].scatter(time_vector,dlm_dt,s = 2)
    ax[1].scatter(time_vector,viscosity,s = 2)
    #ax[0].set_title(r'$\dot{l_{m}}$')
    #ax[1].set_title('Viscosity')
    ax[0].set_ylabel(r'$\dot{l_{m}} \left[\frac{m}{s}\right]$')
    ax[1].set_ylabel(r'Viscosity $\left[\frac{Ns}{m}\right]$')
    ax[1].set_xlabel('Time [s]')
    ax[0].grid(linewidth=0.2)
    ax[1].grid(linewidth=0.2)
    #plt.scatter(test[np.append(np.diff(test) < -0.0001,True)],ld[np.append(np.diff(test) < -0.0001,True)],s = 2, color = 'blue')
    #plt.scatter(test[np.append(np.diff(test) > 0.0001,True)],vc[np.append(np.diff(test) > 0.0001,True)]*test[np.append(np.diff(test) > 0.0001,True)],s = 2, color = 'orange')
    #plt.scatter(test[np.append(np.diff(test) < -0.0001,True)],vc[np.append(np.diff(test) < -0.0001,True)]*test[np.append(np.diff(test) < -0.0001,True)],s = 2, color = 'blue')
    plt.figure()
    plt.plot(time_vector,muscle_force)
    plt.show()


if __name__=="__main__":
    muscle_simulation()
    #muscle_optimization()
    #sigmoid_optimization()
    #rheopectic_simulation()
    #rheopectic_muscle_optimization()
