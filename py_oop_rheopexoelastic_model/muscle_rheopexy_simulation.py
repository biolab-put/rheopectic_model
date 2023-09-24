from rheological_muscle_model import RheopecticMuscle
from muscle_active_force import muscle_active_force
import numpy as np
from matplotlib import pyplot as plt 
from scipy.optimize import differential_evolution
import pandas as pd
import scipy.signal as signal
from scipy.optimize import NonlinearConstraint
import warnings
from scipy.signal import savgol_filter
import time

def load_data(filename):
    data = pd.read_csv(filename)
    return data

def prepare_reference_data_in(filename,sim_dt,steady_time = 0, zeros_time = 0):
    reference_data = load_data(filename)
    reference_data['force'] = reference_data['force'] - reference_data['force'][0] # delete bias
    reference_data['force'] = reference_data['force'] / 1000 # convert from mN to N
    #print(len(reference_data['force']))
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
    return reference_force

def create_rheopectic_model(simulation_dt):
    '''
    differential_evolution step 832: f(x)= 0.247207
    [ 9.27470775e+01  2.69257705e+03  7.47363965e-03  4.18334496e+01
    2.26529378e+01  2.19319711e+01 -2.64438329e+00  1.32691853e+02
    7.25133814e+01  3.85402111e+00  3.24391366e-01  8.31611006e+00
    2.91062475e-01  1.87171287e-02  7.18294449e-02  2.47903488e-03]
    '''
    # Simulation parameters
    sim_dt = simulation_dt
    #act1 = 1 # Active force scaler (input signal scaler)
    F0 = 0 # Initial force
    # Muscle parameters (hill model)
    km =  3.54682634e+00 * 1  #3.44060212e+03 # Muscle stiffness
    kt =  4.48613911e+04 /150 #1.10041022e+05  # Tendon stiffness
    m = 7.27128457e-03 #9.95809312e-03 # Muscle mass
    c_rh_min = 0.5
    c_rh_max = 9.0149924883e+00    #7.22581567e-01# Muscle viscosity
    tr = c_rh_max * 1.15
    G0 = 1.838316377e+00 #15
    delta = 0.005 #7.13479209e-03  #0.0001 # Initial muscle stretch

    # Rheopexy parameters: Two process kinetic model by Mewis and Wagner
    # https://arxiv.org/abs/2212.07572
    k1 = 9.10051262e+01 * 1  #40
    k2 = 4.23556920e+01 * 1.8  #0.5
    A = 3.00518521e+00  #0.5
    B = 4.14698305e-01  #0
    lambda0 = 2.58231526e-01 #0.913209065
    #K = 10.69782781e+00 #0.5#0.5#0.5
    Beta = -3.32562036e+00 #-2#-2 #-2
    # Create the model
    muscle_model = RheopecticMuscle(km,kt,m,c_rh_min,c_rh_max,tr,Beta,k1,k2,lambda0,A,B,G0, delta,sim_dt)
    #muscle_model.set_parameters([ 5.28429299e+03,  5.83971050e+04,  6.68480165e-03,  6.22501928e+01,
  #5.44239417e+01,  3.01253814e+01, -9.51407545e-01,  2.34734230e+02,
  #3.00901059e+02,  1.19263979e-01,  5.22226833e+00,  4.16262706e+01,
  #8.16700863e-01])
    #muscle_model.G0 = muscle_model.G0/10
    
    '''
    muscle_model.set_parameters([9.14506998e+02,  1.78305841e+04,  7.28940956e-03,  0.5, 6.96899335e+01,
  7.26110534e+01, -3.33081384e+00,  4.51368296e+01,
  1.18569616e+02,  4.07441064e+00,  3.28309489e-01,  1.51427411e+01,
  2.64334037e-01])
    '''
    
    '''
    muscle_model.set_parameters([1.36194854e+03  7.86522589e+04  5.39375556e-03  6.80153546e+00
  2.85833778e+01  6.75901342e+01  9.52939885e+00 -5.15120351e+00
  4.93456125e+01  6.40574229e+02  2.87258844e+00  4.00315693e+00
  4.14363968e+01  5.48292729e-01  2.03566800e-02  6.37239622e-02
  3.80066217e-03])
  '''
    #muscle_model.k2 = muscle_model.k2 / 100
    #muscle_model.A = 0
    #muscle_model.B = 0
    #muscle_model.k1 = 0
    #muscle_model.k2 = 0
    #muscle_model.K = 10000
    #muscle_model.Beta = -1
    #muscle_model.tr = 10
    #muscle_model.K = 0
    #muscle_model.Beta = 0
    #muscle_model.c_rh_min = muscle_model.c_rh_min *5
    #muscle_model.lambda0 = 0.1
    #muscle_model.c_rh_min = muscle_model.c_rh_min *2
    #muscle_model.c_rh_max = muscle_model.c_rh_max / 20
    
    '''

    muscle_model.G0 =  muscle_model.G0 *100
    muscle_model.k1 = muscle_model.k1 * 10000
    muscle_model.k2 = muscle_model.k2 / 10000
    muscle_model.A = muscle_model.A
    muscle_model.B = muscle_model.B
    muscle_model.lambda0 = muscle_model.lambda0
    muscle_model.K = 10
    muscle_model.Beta = -10
    '''

    '''
    muscle_model.set_parameters([ 6.72302051e+02,  7.72387498e+03,  9.05419739e-03,  0.25,
  8.86757828e+03,  5.63322811e+01, -7.89866267e+00,  8.92390766e+02,
  6.44572117e+02,  1,  0.6,  5.29830111e+00,
  6.77148652e-01  ])
    '''
    
    '''
    muscle_model.set_parameters([ 6.72302051e+02,  7.72387498e+03,  9.05419739e-03,  0.25,
  8.86757828e+01,  5.63322811e+01, -7.89866267e+00,  3.92390766e+02,
  2.44572117e+02,  0.65,  0.45,  5.29830111e+00,
  6.77148652e-01  ])
  '''
    muscle_model.km = 6.72302051e+02
    muscle_model.kt = 7.72387498e+03
    muscle_model.m = 9.05419739e-03
    muscle_model.c_rh_min = 0.25
    muscle_model.c_rh_max = 8.86757828e+01
    muscle_model.tr = 5.63322811e+01
    muscle_model.Beta = -7.89866267e+00
    muscle_model.k1 = 3.92390766e+02
    muscle_model.k2 = 2.44572117e+04 # zmiana k2 do e+06
    muscle_model.A = 0.55 # 0.53
    muscle_model.B = 0.94 # 1.5
    muscle_model.G0 = 5.29830111e+00 # 15.29830111e+00
    muscle_model.lambda0 = 6.77148652e-01
    
    # Zmiana 1 na 0.45
    return muscle_model

def muscle_identification(muscle_model,time_vector,active_force,reference_force,twitch_duration,twitch_amplitude,twitch_delay,bounds,threads,disp_debug):
    x0 = muscle_model.get_parameters()
    x0.append(twitch_duration)
    x0.append(twitch_amplitude)
    x0.append(twitch_delay)
    #constr = [{'type': 'ineq', 'fun' : lambda x : x[15]/x[14] - 10}]
    constr = NonlinearConstraint(muscle_model.get_stiffness_ratio, 10, 1e09)
    #constr_2 = NonlinearConstraint(muscle_model.get_initial_force, 0.1,0.2)
    #init='sobol', mutation = (0.8,1.99),recombination = 0.5,
    result = differential_evolution(objective,x0 = x0,updating='deferred', args = (muscle_model, time_vector, active_force, reference_force),constraints=(),bounds=bounds,workers = threads, disp = disp_debug,polish=False,callback=print_state)
    #result = differential_evolution(objective,x0 = x0,updating='deferred', args = (muscle_model, time_vector[1], active_force[1], reference_force[1]),constraints=(),bounds=bounds,workers = threads, disp = disp_debug,polish=False,callback=print_state)
    return result

def print_state(xk, convergence):
    print(xk)
    sim_dt = 0.0001
    twitch_delay = xk[-1]
    twitch_amplitude = xk[-2]
    twitch_duration = xk[-3]
    
    
    
    #twitch_delay = 0.0025
    twitch_frequency = 40
    low_frequency = 35
    high_frequency = 75
    #twitch_amplitude = 1
    disp_debug = True
    muscle_model = create_rheopectic_model(sim_dt)

    filenames = ['947kLHLf/947kLHLf_trial_04','947kLHLf/947kLHLf_trial_11','947kLHLf/947kLHLf_trial_17']
    
    reference_force_0 = prepare_reference_data_in(filenames[0],sim_dt,steady_time=0.075,zeros_time=3.5)
    reference_force_1 = prepare_reference_data_in(filenames[1],sim_dt,steady_time=1.5,zeros_time=0.3)
    reference_force_2 = prepare_reference_data_in(filenames[2],sim_dt,steady_time=1.5,zeros_time=0.3)

    #reference_forces = [reference_force_0['force'],reference_force_1['force'],reference_force_2['force']]
    reference_forces = [reference_force_0,reference_force_1,reference_force_2]
    #--------------------------
    #LEN = 18889
    #time_vector_1 = np.arange(0,LEN*sim_dt,sim_dt)
    time_vectors = []
    #time_vectors.append(0)
    #time_vectors.append(0)
    #time_vectors.append(time_vector_1)
    #--------------------------
    time_vector_00 = np.arange(0,len(reference_forces[0]['force'])*sim_dt,sim_dt)
    time_vectors.append(time_vector_00)
    time_vector_0 = np.arange(0,len(reference_forces[1]['force'])*sim_dt,sim_dt)
    time_vectors.append(time_vector_0)
    time_vector_1 = np.arange(0,len(reference_forces[2]['force'])*sim_dt,sim_dt)
    time_vectors.append(time_vector_1)

    
    #input_0 = muscle_active_force.parabolic_twitch(time_vector_00,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    #low_freq_duration = int((19 * ((1/low_frequency - sim_dt/1.5)))/sim_dt)
    
    #high_freq_duration = int((23 * ((1/high_frequency + sim_dt/1.5)))/sim_dt)
    #input_1_low_freq = muscle_active_force.parabolic_twitch(time_vector_1[0:low_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, low_frequency, sim_dt)
    #input_1_high_freq = muscle_active_force.parabolic_twitch(time_vector_1[low_freq_duration:low_freq_duration+high_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, high_frequency, sim_dt)
    #input_1_low_freq_end = muscle_active_force.parabolic_twitch(time_vector_1[low_freq_duration+high_freq_duration:2*low_freq_duration+high_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, low_frequency, sim_dt)

    input_0 = muscle_active_force.parabolic_twitches(reference_forces[2]['stimDig'],twitch_duration,twitch_delay,twitch_amplitude, 0.0001)
    #input_1 = muscle_active_force.parabolic_twitches(reference_forces[1]['stimDig'],twitch_duration,0.0025,1, 0.0001)
    #input_1 = muscle_active_force.parabolic_twitches(reference_forces[2]['stimDig'],twitch_duration,0.0025,1, 0.0001)
    
    #input_1a = muscle_active_force.parabolic_twitch(time_vector_0,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    #input_1a[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal

    #zeros = np.zeros([len(time_vector_1)-(2*low_freq_duration+high_freq_duration)])
    #input_1 = np.concatenate((input_1_low_freq, input_1_high_freq,input_1_low_freq_end,zeros), axis=0)
    #--------------------------
    #active_forces = []
    #active_forces.append(0)
    #active_forces.append(0)
    #active_forces.append(input_1)
    #--------------------------
    #time_vectors = [time_vector_00,time_vector_0,time_vector_1]
    #active_forces = [input_0,input_1a,input_1]

    #reference_stim_dig = np.copy(reference_force_1['stimDig'])
    #reference_stim_dig[reference_stim_dig == 0] = np.nan
    #muscle_model.set_parameters(xk)
    muscle_model.set_parameters(xk[:-3])
    X0 = muscle_model.get_X0()
    #print(xk)
    #fitted_muscle_data_1,[lm, dlm_dt, Lambda] = muscle_model.muscle_response(X0,time_vectors[1],input_1)
    #fitted_muscle_data_1,[lm, dlm_dt,Lambda,rh] = muscle_model.muscle_response(X0,time_vectors[0],input_0)
    fitted_muscle_data_1,[lm,dlm_dt, Lambda,rh] = muscle_model.muscle_response(X0,time_vectors[2],input_0)

    '''
    filtered_reference = savgol_filter(reference_forces[2]['force'],600,1)
    filtered_estimated_force = savgol_filter(fitted_muscle_data_1,600,1)
    forces_error = np.sum(np.square(filtered_reference - filtered_estimated_force))
    reference_force_derative = savgol_filter(np.gradient(filtered_reference,muscle_model.sim_dt),600,1)
    filtered_estimated_force_derative = savgol_filter(np.gradient(filtered_estimated_force,muscle_model.sim_dt),600,1)
    forces_derative_error = 0.01 * np.sum(np.square(reference_force_derative - filtered_estimated_force_derative))

    reference_force_std_window = np.lib.stride_tricks.sliding_window_view(reference_forces[2]['force'], 100)
    reference_force_mstd = savgol_filter(np.std(reference_force_std_window, axis=-1),250,3)

    estimated_force_std_window = np.lib.stride_tricks.sliding_window_view(fitted_muscle_data_1, 100)
    estimated_force_mstd = savgol_filter(np.std(estimated_force_std_window, axis=-1),250,3)
    std_forces_error = 10 * np.sum(np.square(reference_force_mstd - estimated_force_mstd))

    estimated_force_mstd_derative = np.gradient(estimated_force_mstd,muscle_model.sim_dt)
    reference_force_mstd_derative = np.gradient(reference_force_mstd,muscle_model.sim_dt)
    std_forces_derative_error = 0.001 * np.sum(np.square(reference_force_mstd_derative - estimated_force_mstd_derative))

    error = forces_error + forces_derative_error + std_forces_error + std_forces_derative_error
    print('Forces error', forces_error)
    print('Forces derative error', forces_derative_error)
    print('Std forces error', std_forces_error)
    print('Std forces derative error', std_forces_derative_error)
    print('Cumulative error', error)
    '''

    #filtered_reference = savgol_filter(reference_forces[2]['force'],600,1)
    #filtered_estimated_force = savgol_filter(fitted_muscle_data_1,600,1)

    #reference_force_derative = np.gradient(filtered_reference,muscle_model.sim_dt)
    #filtered_estimated_force_derative = np.gradient(filtered_estimated_force,muscle_model.sim_dt)

    #forces_error = np.sum(np.square(filtered_reference - filtered_estimated_force))
    #forces_error_transient = 10 * np.sum(np.square(filtered_reference[int(0.95/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_estimated_force[int(0.95/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)]))
    #print('Forces error', forces_error)
    #print('Forces error transient', forces_error_transient)

    #derative_error = 0.01 * np.sum(np.square(reference_force_derative - filtered_estimated_force_derative))
    #derative_error_transient = 0.1 * np.sum(np.square(reference_force_derative[int(0.95/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_estimated_force_derative[int(0.95/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)]))
    #print('Derative error', derative_error)
    #print('Derative error transient', derative_error_transient)

    #print('Cumulative error', forces_error + forces_error_transient + derative_error + derative_error_transient)


    #axs.plot(time_vectors[1][0:int(0.95/muscle_model.sim_dt)], fitted_muscle_data_1[0:int(0.95/muscle_model.sim_dt)])
    #axs.plot(time_vectors[1][int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)], fitted_muscle_data_1[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])
    #axs.plot(time_vectors[1][int(1.395/muscle_model.sim_dt)::], fitted_muscle_data_1[int(1.395/muscle_model.sim_dt)::])

    #axs.plot(time_vectors[1][0:int(0.95/muscle_model.sim_dt)], reference_forces[1]['force'][0:int(0.95/muscle_model.sim_dt)])
    #axs.plot(time_vectors[1][int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)], reference_forces[1]['force'][int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])
    #axs.plot(time_vectors[1][int(1.395/muscle_model.sim_dt)::], reference_forces[1]['force'][int(1.395/muscle_model.sim_dt)::])
    
    plt.close('all')
    fig, axs = plt.subplots(1)
    axs.plot(time_vectors[2], fitted_muscle_data_1)
    axs.plot(time_vectors[2], reference_forces[2]['force'])
    plt.pause(0.0001)
    


def objective(x,muscle_model,time_vector, active_force,reference_force):
    error = 0
    twitch_delay = x[-1]
    twitch_amplitude = x[-2]
    twitch_duration = x[-3]
    model_parameters = x[:-3]
    #model_parameters = x

    muscle_model.set_parameters(model_parameters)
    X0 = muscle_model.get_X0()
    for i in range(len(reference_force)):
        if i !=0:
            input_1 = muscle_active_force.parabolic_twitches(reference_force[i]['stimDig'],twitch_duration,twitch_delay,twitch_amplitude, 0.0001)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                muscle_data,_ = muscle_model.muscle_response(X0,time_vector[i],input_1)

                #filtered_reference = savgol_filter(reference_force,600,1)
                #filtered_reference = savgol_filter(reference_force['force'],600,1)
                #filtered_estimated_force = savgol_filter(muscle_data,600,1)

                #filtered_reference = savgol_filter(reference_force['force'],600,1)
                filtered_reference = savgol_filter(reference_force[i]['force'],600,1)
                filtered_estimated_force = savgol_filter(muscle_data,600,1)

                #reference_force_derative = np.gradient(filtered_reference,muscle_model.sim_dt)
                #filtered_estimated_force_derative = np.gradient(filtered_estimated_force,muscle_model.sim_dt)
                #error += np.sum(np.square(reference_force - muscle_data))
                test  = 0
                if( i == 1):
                    error += 1* np.sum(np.square(filtered_reference - filtered_estimated_force))
                    reference_force_std_window = np.lib.stride_tricks.sliding_window_view(reference_force[i]['force'], 100)
                    reference_force_mstd = savgol_filter(np.std(reference_force_std_window, axis=-1),250,3)

                    estimated_force_std_window = np.lib.stride_tricks.sliding_window_view(muscle_data, 100)
                    estimated_force_mstd = savgol_filter(np.std(estimated_force_std_window, axis=-1),250,3)
                    std_forces_error = 10 * np.sum(np.square(reference_force_mstd - estimated_force_mstd))
                    #error += np.sum(np.square(filtered_reference[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_estimated_force[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)]))
                    error += std_forces_error
                    
                    error += 40 * np.sum(np.square(filtered_reference[int(0.9/muscle_model.sim_dt):int(1.0/muscle_model.sim_dt)] - filtered_estimated_force[int(0.9/muscle_model.sim_dt):int(1.0/muscle_model.sim_dt)]))
                    #print('Err 1: ', error)
                    #error += 40 * np.sum(np.square(filtered_reference[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_estimated_force[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)]))
                    #error += 85 * np.sum(np.square(reference_force[i]['force'][0:int(0.04/muscle_model.sim_dt)] - muscle_data[0:int(0.04/muscle_model.sim_dt)]))d
                if( i == 2):
                    
                    error += np.sum(np.square(filtered_reference - filtered_estimated_force))
                    
                    reference_force_std_window = np.lib.stride_tricks.sliding_window_view(reference_force[i]['force'], 100)
                    reference_force_mstd = savgol_filter(np.std(reference_force_std_window, axis=-1),250,3)

                    estimated_force_std_window = np.lib.stride_tricks.sliding_window_view(muscle_data, 100)
                    estimated_force_mstd = savgol_filter(np.std(estimated_force_std_window, axis=-1),250,3)
                    std_forces_error = 10 * np.sum(np.square(reference_force_mstd - estimated_force_mstd))
                    #error += np.sum(np.square(filtered_reference[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_estimated_force[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)]))
                    error += std_forces_error

                    error += 50 * np.sum(np.square(filtered_reference[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_estimated_force[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)]))
                    #error += 85/1.2 * np.sum(np.square(reference_force[i]['force'][0:int(0.04/muscle_model.sim_dt)] - muscle_data[0:int(0.04/muscle_model.sim_dt)]))
                #print('Err 2: ', error - temp_err)
    #input_1 = muscle_active_force.parabolic_twitches(reference_force['stimDig'],twitch_duration,twitch_delay,twitch_amplitude, 0.0001)

    '''
    muscle_model.set_parameters(model_parameters)
    X0 = muscle_model.get_X0()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        muscle_data,_ = muscle_model.muscle_response(X0,time_vector,input_1)

        #filtered_reference = savgol_filter(reference_force,600,1)
        #filtered_reference = savgol_filter(reference_force['force'],600,1)
        #filtered_estimated_force = savgol_filter(muscle_data,600,1)

        #filtered_reference = savgol_filter(reference_force['force'],600,1)
        filtered_reference = savgol_filter(reference_force['force'],600,1)
        filtered_estimated_force = savgol_filter(muscle_data,600,1)

        #reference_force_derative = np.gradient(filtered_reference,muscle_model.sim_dt)
        #filtered_estimated_force_derative = np.gradient(filtered_estimated_force,muscle_model.sim_dt)
        #error += np.sum(np.square(reference_force - muscle_data))

        error += np.sum(np.square(filtered_reference - filtered_estimated_force))
        reference_force_std_window = np.lib.stride_tricks.sliding_window_view(reference_force['force'], 100)
        reference_force_mstd = savgol_filter(np.std(reference_force_std_window, axis=-1),250,3)

        estimated_force_std_window = np.lib.stride_tricks.sliding_window_view(muscle_data, 100)
        estimated_force_mstd = savgol_filter(np.std(estimated_force_std_window, axis=-1),250,3)
        std_forces_error = 10 * np.sum(np.square(reference_force_mstd - estimated_force_mstd))
        error += np.sum(np.square(filtered_reference[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_estimated_force[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)]))
        error += std_forces_error

        error += 40 * np.sum(np.square(filtered_reference[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_estimated_force[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)]))
        error += 85 * np.sum(np.square(reference_force['force'][0:int(0.04/muscle_model.sim_dt)] - muscle_data[0:int(0.04/muscle_model.sim_dt)]))

        #error += np.sum(np.square(reference_force['force'] - muscle_data))

        #print('A', np.sum(np.square(filtered_reference - filtered_estimated_force)))
        #error += np.sum(np.square(filtered_reference[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_estimated_force[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)]))
        #print('B', np.sum(np.square(filtered_reference[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_estimated_force[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)])))
        
        #error += 0.1 * np.sum(np.square(reference_force_derative - filtered_estimated_force_derative))
        #error += np.sum(np.square(reference_force_derative[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_estimated_force_derative[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)]))
        #print('C', 0.1 * np.sum(np.square(reference_force_derative - filtered_estimated_force_derative)))
        #print('D', np.sum(np.square(reference_force_derative[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_estimated_force_derative[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)])))
    #B, A = signal.butter(2, 0.003, output='ba')
    #filtered_reference = signal.filtfilt(B,A,reference_force)
    #filtered_estimated_force= signal.filtfilt(B,A,muscle_data)
    #reference_force_derative = np.gradient(filtered_reference,muscle_model.sim_dt)
    #estimated_force_derative = np.gradient(filtered_estimated_force,muscle_model.sim_dt)


    #error += np.sum(np.abs(reference_force - muscle_data))
    #error += 10 * np.sum(np.abs(reference_force[int(1.05/muscle_model.sim_dt):int(1.5/muscle_model.sim_dt)] - muscle_data[int(1.05/muscle_model.sim_dt):int(1.5/muscle_model.sim_dt)]))

    #error += 5 * np.sum(np.abs(reference_force[int(0.0/muscle_model.sim_dt):int(0.25/muscle_model.sim_dt)] - muscle_data[int(0.0/muscle_model.sim_dt):int(0.25/muscle_model.sim_dt)]))
    #print(error)
    #print(np.mean(muscle_data))
    #print(model_parameters)
    '''
    return error

def rheopectic_modified_muscle_optimization():
    sim_dt = 0.0001

         
    twitch_duration =  2.13684632e-02#1.93976525e-02#2.13684632e-02#1.94042729e-02 #2.05572168e-02  #1.53709415e-02  #2.06812073e-02 #2.05572168e-02#2.05572168e-02#1.94042729e-02 # #1.87171287e-02#1.08064681e-02 ##7.43693943e-03#1.87171287e-02 #1.34594494e-02  #1.87171287e-02 #1.806677132e-02 #1.38720902e-02 #5.00338638e-03
    twitch_delay = 5.41840492e-03#2.71686679e-03#5.41840492e-03#2.08183690e-03#2.05624797e-03#4.82148830e-03#2.04295978e-03#2.05624797e-03#2.05624797e-03#2.08183690e-03##2.47903488e-03#4.79517954e-03##3.30621295e-03#2.47903488e-03#5.84467490e-03#2.47903488e-03 #2.39753708e-03 #2.17465697e-03 #2.08550640e-03
    twitch_frequency = 40 #input 1a
    low_frequency = 35 #input 1
    high_frequency = 75
    twitch_amplitude =  8.62972358e-02 #7.08897978e-02  #8.62972358e-02#6.99678867e-02  #6.80343112e-02  #1.20664179e-01#6.48254423e-02  #6.80343112e-02 #6.80343112e-02#6.99678867e-02 # #7.18294449e-02#1.38797015e-01 ##1.23989518e+00#7.18294449e-02#1.33077709e+00#7.18294449e-02 #1/14 #1.06935107e-01 #1.34342082e-01
    disp_debug = True
    muscle_model = create_rheopectic_model(sim_dt)
    threads = 8
    X0 = muscle_model.get_X0()
    c_rh_min_bound = (0.1,8)
    c_rh_max_bound = (10,150)
    k1_bound = (0.001,900)
    k2_bound = (0.001,900)
    lambda0_bound = (0.1,1)
    km_bound = (0.01,10000)
    kt_bound = (0.01,100000)
    m_bound = (0.0001, 0.01)
    A_bound = (0,7)
    B_bound = (0,7)
    tr_bound = (0.0001,80) # (0.01,200)
    #act1_bound = (muscle_model.act1,muscle_model.act1)
    #K_bound = (0.01,50)
    Beta_bound = (-20,-0.01)
    delta_bound = (0.00001,0.01)
    twitch_duration_bound = (0.005,0.022)
    twitch_amplitude_bound = (0.001,2)
    twitch_delay_bound = (0.002,0.006)
    G0_bound = (0.01,25)
    rh_lambda_b_bound = (0.4,0.8)
    rh_lambda_c_bound = (0.1,0.2)
    bounds = (km_bound,kt_bound,m_bound,c_rh_min_bound,c_rh_max_bound,tr_bound,Beta_bound,k1_bound,k2_bound,A_bound,B_bound,G0_bound,lambda0_bound,twitch_duration_bound,twitch_amplitude_bound,twitch_delay_bound)
    #bounds = (rh_lambda_b_bound,rh_lambda_c_bound,(2.13684631e-02,2.13684633e-02),(8.62972357e-02,8.62972359e-02),(5.41840491e-03,5.41840493e-03))
    damping_ratio = 1.
    damping_ratio_margin = 0.1

    # Prepare reference data
    # ----------------------
    X0 = muscle_model.get_X0()
    # load reference data
    filenames = ['947kLHLf/947kLHLf_trial_04','947kLHLf/947kLHLf_trial_11','947kLHLf/947kLHLf_trial_17']
    #filenames = ['1006j_trial_04','1006j_trial_09','1006j_trial_15']
    
    reference_force_0 = prepare_reference_data_in(filenames[0],sim_dt,steady_time=0.075,zeros_time=3.5)
    reference_force_1 = prepare_reference_data_in(filenames[1],sim_dt,steady_time=1.5,zeros_time=0.3)
    reference_force_2 = prepare_reference_data_in(filenames[2],sim_dt,steady_time=1.5,zeros_time=0.3)


    reference_forces = [reference_force_0,reference_force_1,reference_force_2]
    reference_forces[2]['force'][0:int(twitch_delay/sim_dt)] = 0
    reference_forces[2]['force'] = np.clip(reference_forces[2]['force'],0,np.inf)
    #print(int(twitch_delay/sim_dt))


    time_vector_00 = np.arange(0,len(reference_forces[0]['force'])*sim_dt,sim_dt)
    
    time_vector_0 = np.arange(0,len(reference_forces[1]['force'])*sim_dt,sim_dt)
    time_vector_1 = np.arange(0,len(reference_forces[2]['force'])*sim_dt,sim_dt)
    time_vector_1 = np.linspace(0, len(reference_forces[2]['force'])*sim_dt-sim_dt, len(reference_forces[2]['force']))

    input_0 = muscle_active_force.parabolic_twitches(reference_forces[0]['stimDig'],twitch_duration,twitch_delay,twitch_amplitude, sim_dt)
    input_1a = muscle_active_force.parabolic_twitches(reference_forces[1]['stimDig'],twitch_duration,twitch_delay,twitch_amplitude, sim_dt)
    input_1 = muscle_active_force.parabolic_twitches(reference_forces[2]['stimDig'],twitch_duration,twitch_delay,twitch_amplitude, sim_dt)

    time_vectors = [time_vector_00,time_vector_0,time_vector_1]
    active_forces = [input_0,input_1a,input_1]

    reference_stim_dig = np.copy(reference_force_2['stimDig'])
    reference_stim_dig[reference_stim_dig == 0] = np.nan

    print(np.array(1/np.diff(time_vectors[1][reference_forces[1]['stimDig'] == 1])))
    print(np.unique(np.array(1/np.diff(time_vectors[1][reference_forces[1]['stimDig'] == 1]),dtype=np.uint8)))


    #plt.plot(time_vectors[2], reference_forces[2]['force'])
    #plt.show()
    #exit()
    # prepare input data

    # ----------------------
    
    # Get muscle response

    #muscle_model.set_parameters([1.70106481e+01, 1.79950186e+02, 6.91542567e-03, 5.14409332e+02])



    X0 = muscle_model.get_X0()
    st = time.time()
    muscle_data,[lm,dlm_dt, Lambda,rh] = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1])
    print('Time', time.time()-st)
    #print(time_vectors[2].shape)
    
    
    #print(time.time() - st)

    #st = time.time()
    #muscle_data,[lm,dlm_dt, Lambda,rh]  = muscle_model.muscle_response(X0,time_vectors[2],active_forces[2])
    #print(time.time() - st)
    #plt.plot(time_vectors[2],Lambda)
    #plt.show()
    #exit()
    #filtered_reference = savgol_filter(reference_forces[2]['force'],600,1)
    #filtered_estimated_force = savgol_filter(muscle_data,600,1)
    
    #print(np.sum(np.square(filtered_reference - filtered_estimated_force)))


    #reference_force_std_window = np.lib.stride_tricks.sliding_window_view(reference_forces[2]['force'], 100)
    #reference_force_mstd = savgol_filter(np.std(reference_force_std_window, axis=-1),250,3)

    #estimated_force_std_window = np.lib.stride_tricks.sliding_window_view(muscle_data, 100)
    #estimated_force_mstd = savgol_filter(np.std(estimated_force_std_window, axis=-1),250,3)
    #std_forces_error = 10 * np.sum(np.square(reference_force_mstd - estimated_force_mstd))


    #print(std_forces_error)
    #print(80 * np.sum(np.square(filtered_reference[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_estimated_force[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)])))
    #print(10 * np.sum(np.square(filtered_reference[0:int(0.065/muscle_model.sim_dt)] - filtered_estimated_force[0:int(0.065/muscle_model.sim_dt)])))
    #exit()


    #muscle_data,[lm,dlm_dt] = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1])
    #a = time.time()


    #muscle_data,[lm,dlm_dt] = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1])
    #print(time.time() - a)
    #exit()
    #print(lm)

    
    fig, axs = plt.subplots(2)
    axs[0].plot(time_vectors[1], muscle_data)
    axs[0].plot(time_vectors[1], reference_forces[1]['force'])
    axs[0].set_title('Muscle force 1')
    axs[1].plot(time_vectors[1], Lambda)
    muscle_data,[lm,dlm_dt, Lambda,rh] = muscle_model.muscle_response(X0,time_vectors[2],active_forces[2])
    fig_1, axs_1 = plt.subplots(2)
    axs_1[0].plot(time_vectors[2], muscle_data)
    axs_1[0].plot(time_vectors[2], reference_forces[2]['force'])
    axs_1[0].set_title('Muscle force 2')
    axs_1[1].plot(time_vectors[2], Lambda)
    plt.show()
    exit()
    
    
    '''
    # Plot results
    plt.figure()
    #plt.plot(time_vectors[1], rh)
    plt.plot(time_vectors[1], muscle_data)
    plt.plot(time_vectors[1], reference_forces[1]['force'])
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle force [N]')
    #plt.ylim([0.97,1])
    plt.grid()
    plt.legend(['Estimated force 1','Reference force 1'])
    #plt.show()

    muscle_data,[lm,dlm_dt, Lambda,rh] = muscle_model.muscle_response(X0,time_vectors[2],active_forces[2])

    plt.figure()
    #plt.ylim([0.97,1])
    #plt.plot(time_vectors[2], rh)
    plt.plot(time_vectors[2], muscle_data)
    plt.plot(time_vectors[2], reference_forces[2]['force'])
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle force [N]')
    plt.grid()
    plt.legend(['Estimated force 2','Reference force 2'])
    plt.show()
    exit()
    '''

    # Muscle parameter search (optional)
    '''
    p = muscle_model.get_parameters()
    for i in range(len(p)):
        print(i, len(p),p[i])
        p[i] *= 5
        #print(p[0],p[i])
        muscle_model.set_parameters(p)
        X0 = muscle_model.get_X0()
        muscle_data,[lm,dlm_dt, Lambda] = muscle_model.muscle_response(X0,time_vectors[2],active_forces[2])
        B, A = signal.butter(2, 0.003, output='ba')
        filtered_reference = signal.filtfilt(B,A,reference_forces[2])
        filtered_estimated_force= signal.filtfilt(B,A,muscle_data)
        reference_force_derative = np.gradient(filtered_reference,muscle_model.sim_dt)
        estimated_force_derative = np.gradient(filtered_estimated_force,muscle_model.sim_dt)
        plt.plot(time_vectors[2], estimated_force_derative)
        plt.plot(time_vectors[2], reference_force_derative)
        plt.xlabel('Time [s]')
        plt.ylabel('Muscle force [N]')
        plt.grid()
        plt.legend(['Estimated force','Reference force'])
        plt.figure()
        plt.plot(time_vectors[2], muscle_data)
        plt.plot(time_vectors[2], reference_forces[2])
        plt.show()
        p[i] /= 10
        muscle_model.set_parameters(p)
    exit()
    '''

    # Optimize
    result = muscle_identification(muscle_model,time_vectors,active_forces,reference_forces,twitch_duration,twitch_amplitude,twitch_delay,bounds,threads,disp_debug)

    active_forces[2] = muscle_active_force.parabolic_twitches(reference_forces[2]['stimDig'],result.x[-3],result.x[-1],result.x[-2], sim_dt)
    muscle_model.set_parameters(result.x[:-3])
    X0 = muscle_model.get_X0()
    #muscle_data,[lm,dlm_dt, Lambda,rh] = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1])
    muscle_data,[lm,dlm_dt,Lambda,rh] = muscle_model.muscle_response(X0,time_vectors[2],active_forces[2])

    print(result)
    plt.plot(time_vectors[2], muscle_data)
    plt.plot(time_vectors[2], reference_forces[2]['force'])
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle force [N]')
    plt.grid()
    plt.legend(['Estimated force','Reference force'])
    plt.show()
    exit()
    fitted_muscle_data_00,_ = muscle_model.muscle_response(X0,time_vectors[0],active_forces[0])
    fitted_muscle_data_0,_ = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1])
    fitted_muscle_data_1,_ = muscle_model.muscle_response(X0,time_vectors[2],active_forces[2])

    B, A = signal.butter(2, 0.012, output='ba')

    plt.figure(0)
    plt.plot(time_vectors[0], reference_forces[0])
    plt.plot(time_vectors[0], signal.filtfilt(B,A, reference_forces[0]))
    plt.plot(time_vectors[0], fitted_muscle_data_00)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')

    plt.figure(1)
    plt.plot(time_vectors[1], reference_forces[1])
    plt.plot(time_vectors[1], signal.filtfilt(B,A, reference_forces[1]))
    plt.plot(time_vectors[1], fitted_muscle_data_0)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.figure(2)
    plt.plot(time_vectors[2], reference_forces[2])
    plt.plot(time_vectors[2], signal.filtfilt(B,A, reference_forces[2]))
    plt.plot(time_vectors[2], fitted_muscle_data_1)
    reference_stim_dig = np.copy(reference_force_2['stimDig'])
    reference_stim_dig[reference_stim_dig == 0] = np.nan
    plt.scatter(time_vectors[2],reference_stim_dig * 0,2,'black',alpha = 1)
    plt.grid(color='grey', linestyle='-', linewidth=0.3)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.show()

def main():
    rheopectic_modified_muscle_optimization()

if __name__=="__main__":
    main()
