from rheological_muscle_model import RheopecticMuscle
from muscle_active_force import muscle_active_force
import numpy as np
from matplotlib import pyplot as plt 
from scipy.optimize import differential_evolution
import pandas as pd
import scipy.signal as signal

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
    return reference_force

def create_rheopectic_model(simulation_dt):
    sim_dt = simulation_dt
    km = 8.21688727e+00 / 100 #* 400 #  * 1.1 
    kt = 5.45036325e+01 * 100 #* 50 # / 1.2
    m = 2.86717809e-02 
    cs = 1.35102053e+01 * 2 
    ks = 2.45926152e+02 * 1 
    ls0 = -0.0001
    delta = 0.01
    k1 = 2.79147980e+02  / 15 #* 1.5
    k2 = 6.98119462e+03 * 5
    c_rh = 2.10558506e+01 
    c_rh_min = 1.0490643e+00 
    c1 = 2.30490643e+01 
    lambda0 = 3.13209065e-01
    F0 = 0.1
    s1 = 1

    muscle_model = RheopecticMuscle(km,kt,m,cs,ks,ls0,c_rh,c_rh_min,c1,k1,k2,lambda0,F0,s1,delta,sim_dt)
    return muscle_model


def rheopectic_modified_muscle_optimization():
    sim_dt = 0.0001
    twitch_duration = 0.01
    twitch_delay = 0.0025
    twitch_frequency = 40 #input 1a
    low_frequency = 35 #input 1
    high_frequency = 75
    twitch_amplitude = 1
    disp_debug = True

    muscle_model = create_rheopectic_model(sim_dt)

    
    c_rh_bound = (muscle_model.c_rh/10,muscle_model.c_rh*10)
    c1_bound = (muscle_model.c1/10,muscle_model.c1*10)
    k1_bound = (muscle_model.k1/10,muscle_model.k1*10)
    k2_bound = (muscle_model.k2/10,muscle_model.k2*10)
    lambda0_bound = (0.1,1)
    cs_bound = (muscle_model.cs/10,muscle_model.cs*10)
    c1_bound = (muscle_model.c1/10,muscle_model.c1*10)
    ks_bound = (muscle_model.ks/10,muscle_model.ks*10)
    c_rh_min_bound = (muscle_model.c_rh_min/10,muscle_model.c_rh_min*10)
    km_bound = (muscle_model.km/10,muscle_model.km*10)
    kt_bound = (muscle_model.kt/10,muscle_model.kt*10)
    
    ls0_bound = (muscle_model.ls0/10,muscle_model.ls0*10)
    F0_bound = (0,1)
    km_bound = (0.1,5000)
    kt_bound = (0.1,5000)
    s1_bound = (0.00001,10000)
    m_bound = (0.0001,0.01)
    bounds = (km_bound,kt_bound,m_bound,cs_bound,ks_bound,ls0_bound,c_rh_bound,c_rh_min_bound,c1_bound,k1_bound,k2_bound,lambda0_bound,F0_bound,s1_bound)
    threads = 12
    damping_ratio = 1.
    damping_ratio_margin = 0.1

    # Prepare reference data
    # ----------------------
    X0 = muscle_model.get_X0()
    # load reference data
    filenames = ['947kLHLf/947kLHLf_trial_04','947kLHLf/947kLHLf_trial_07','947kLHLf/947kLHLf_trial_17']
    
    reference_force_0 = prepare_reference_data_in(filenames[0],sim_dt,steady_time=0.075,zeros_time=3.5)
    reference_force_1 = prepare_reference_data_in(filenames[1],sim_dt,steady_time=1.5,zeros_time=0.3)
    reference_force_2 = prepare_reference_data_in(filenames[2],sim_dt,steady_time=1.5,zeros_time=0.3)

    reference_forces = [reference_force_0['force'],reference_force_1['force'],reference_force_2['force']]
    time_vector_00 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    time_vector_0 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    time_vector_1 = np.arange(0,len(reference_forces[2])*sim_dt,sim_dt)

    # prepare input data
    input_0 = muscle_active_force.parabolic_twitch(time_vector_00,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    low_freq_duration = int((19 * ((1/low_frequency - sim_dt/1.5)))/sim_dt)
    
    high_freq_duration = int((23 * ((1/high_frequency + sim_dt/1.5)))/sim_dt)
    input_1_low_freq = muscle_active_force.parabolic_twitch(time_vector_1[0:low_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, low_frequency, sim_dt)
    input_1_high_freq = muscle_active_force.parabolic_twitch(time_vector_1[low_freq_duration:low_freq_duration+high_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, high_frequency, sim_dt)
    input_1_low_freq_end = muscle_active_force.parabolic_twitch(time_vector_1[low_freq_duration+high_freq_duration:2*low_freq_duration+high_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, low_frequency, sim_dt)

    input_1a = muscle_active_force.parabolic_twitch(time_vector_0,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    input_1a[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal

    zeros = np.zeros([len(time_vector_1)-(2*low_freq_duration+high_freq_duration)])
    input_1 = np.concatenate((input_1_low_freq, input_1_high_freq,input_1_low_freq_end,zeros), axis=0)
    time_vectors = [time_vector_00,time_vector_0,time_vector_1]
    active_forces = [input_0,input_1a,input_1]

    reference_stim_dig = np.copy(reference_force_2['stimDig'])
    reference_stim_dig[reference_stim_dig == 0] = np.nan
    # ----------------------
    
    # Get muscle response
    muscle_data,[lm,dlm_dt, Lambda,ls] = muscle_model.muscle_response(X0,time_vectors[2],active_forces[2])

    # Plot results
    B, A = signal.butter(2, 0.012, output='ba')
    filtered_reference = signal.filtfilt(B,A,reference_forces[2])
    filtered_estimated_force= signal.filtfilt(B,A,muscle_data)
    reference_force_derative = np.gradient(filtered_reference,muscle_model.sim_dt)
    estimated_force_derative = np.gradient(filtered_estimated_force,muscle_model.sim_dt)
    B, A = signal.butter(2, 0.003, output='ba')
    filtered_reference = signal.filtfilt(B,A,reference_forces[2])
    reference_force_derative = np.gradient(filtered_reference,muscle_model.sim_dt)
    
    
    '''
    plt.plot(time_vectors[2][0:int(0.16/muscle_model.sim_dt)], muscle_data[0:int(0.16/muscle_model.sim_dt)])
    plt.plot(time_vectors[2][int(0.16/muscle_model.sim_dt):int(0.95/muscle_model.sim_dt)], muscle_data[int(0.16/muscle_model.sim_dt):int(0.95/muscle_model.sim_dt)])
    plt.plot(time_vectors[2][0:int(0.95/muscle_model.sim_dt)], muscle_data[0:int(0.95/muscle_model.sim_dt)])
    plt.plot(time_vectors[2][int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)], muscle_data[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])
    plt.plot(time_vectors[2][int(1.395/muscle_model.sim_dt)::], muscle_data[int(1.395/muscle_model.sim_dt)::])

    plt.plot(time_vectors[2][0:int(0.95/muscle_model.sim_dt)], reference_forces[2][0:int(0.95/muscle_model.sim_dt)])
    plt.plot(time_vectors[2][int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)], reference_forces[2][int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])
    plt.plot(time_vectors[2][int(1.395/muscle_model.sim_dt)::], reference_forces[2][int(1.395/muscle_model.sim_dt)::])
    '''
    plt.plot(time_vectors[2], muscle_data)
    plt.plot(time_vectors[2], reference_forces[2])

    #plt.plot(Lambda)
    #plt.figure()
    #plt.plot(reference_force_2['stimDig'])
    #plt.plot(active_forces[2])
    #viscosity = muscle_model.c_rh * Lambda * dlm_dt + muscle_model.c_rh_min * dlm_dt
    #plt.plot(viscosity)
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle force [N]')
    plt.grid()
    plt.legend(['Estimated force','Reference force'])
    plt.show()
    exit()
    

    # Optimize
    twitch_data = (twitch_duration,twitch_amplitude)
    result = muscle_identification(muscle_model,time_vectors,active_forces,reference_forces,twitch_data,damping_ratio,damping_ratio_margin,bounds,threads,disp_debug)
    #result = muscle_grid_search(muscle_model,time_vectors,active_forces,reference_forces,bounds,threads,disp_debug)
    muscle_model.set_parameters(result.x)
    X0 = muscle_model.get_X0()
    print(result)
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