from re import T
from time import time
from turtle import color

from sympy import true
from hill_muscle_model import hill_muscle_model
from hill_muscle_model import rheopectic_hill_muscle_model
from hill_muscle_model import modified_hill_muscle_model
from hill_muscle_model import rheopectic_modified_hill_muscle_model
from muscle_active_force import muscle_active_force
import numpy as np
from matplotlib import pyplot as plt 
from scipy.optimize import NonlinearConstraint
from scipy.optimize import differential_evolution
from scipy.optimize import brute
import pandas as pd
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from scipy import fftpack
import scipy.signal as signal



#plt.show()

def find_zero_crossing(data,nearest_point,tolerance):
    zeros = np.diff(np.sign(data))
    index = np.where(zeros)[0]
    index_t = np.where(np.isclose(index,np.ones(np.shape(index),dtype=int) * int(nearest_point),rtol = 0,atol = tolerance) == True)[0]
    return index[index_t]

def check_signum(data,nearest_point,window_size):
    data_left = data[nearest_point-window_size:nearest_point]
    data_right = data[nearest_point+1:nearest_point+window_size+1]

    if(np.all(data_left < 0 ) and  np.all(data_right > 0 )):
        return True,[np.logical_not(data_left < 0), np.logical_not(data_right > 0) ]
    return False,[np.logical_not(data_left < 0), np.logical_not(data_right > 0) ]
   

def print_state(xk, convergence):
    sim_dt = 0.0001
    twitch_duration = 0.01
    twitch_delay = 0.0025
    twitch_frequency = 40
    low_frequency = 35
    high_frequency = 75
    twitch_amplitude = 1
    disp_debug = True
    muscle_model = create_rheopectic_modified_model(sim_dt)

    filenames = ['947kLHLf/947kLHLf_trial_04','947kLHLf/947kLHLf_trial_07','947kLHLf/947kLHLf_trial_17']
    
    reference_force_0 = prepare_reference_data_in(filenames[0],sim_dt,steady_time=0.075,zeros_time=3.5)
    reference_force_1 = prepare_reference_data_in(filenames[1],sim_dt,steady_time=1.5,zeros_time=0.3)
    reference_force_2 = prepare_reference_data_in(filenames[2],sim_dt,steady_time=1.5,zeros_time=0.3)

    reference_forces = [reference_force_0['force'],reference_force_1['force'],reference_force_2['force']]
    time_vector_00 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    time_vector_0 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    time_vector_1 = np.arange(0,len(reference_forces[2])*sim_dt,sim_dt)

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

    reference_stim_dig = np.copy(reference_force_1['stimDig'])
    reference_stim_dig[reference_stim_dig == 0] = np.nan
    muscle_model.set_parameters(xk)
    X0 = muscle_model.get_X0()
    print(xk)
    #fitted_muscle_data_00,_ = muscle_model.muscle_response(X0,time_vectors[0],active_forces[0])
    #fitted_muscle_data_0,_ = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1])
    fitted_muscle_data_1,[lm, dlm_dt, Lambda,ls] = muscle_model.muscle_response(X0,time_vectors[2],active_forces[2])
    #B, A = signal.butter(3, 0.003, output='ba')

    '''
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
    '''
    #plt.close('all')
    #plt.ion()
    #plt.figure(2)
    #plt.show()
    plt.close('all')
    fig, axs = plt.subplots(1)
    #axs[0].plot(time_vectors[2], reference_forces[2])
    #axs[0].plot(time_vectors[2], signal.filtfilt(B,A, reference_forces[2]))
    #axs[0].plot(time_vectors[2], fitted_muscle_data_1)
    
    axs.plot(time_vectors[2][0:int(0.95/muscle_model.sim_dt)], fitted_muscle_data_1[0:int(0.95/muscle_model.sim_dt)])
    axs.plot(time_vectors[2][int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)], fitted_muscle_data_1[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])
    axs.plot(time_vectors[2][int(1.395/muscle_model.sim_dt)::], fitted_muscle_data_1[int(1.395/muscle_model.sim_dt)::])

    axs.plot(time_vectors[2][0:int(0.95/muscle_model.sim_dt)], reference_forces[2][0:int(0.95/muscle_model.sim_dt)])
    axs.plot(time_vectors[2][int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)], reference_forces[2][int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])
    axs.plot(time_vectors[2][int(1.395/muscle_model.sim_dt)::], reference_forces[2][int(1.395/muscle_model.sim_dt)::])
    #plt.show(block = False)
    plt.pause(0.0001)

    '''
    B, A = signal.butter(2, 0.003, output='ba')
    filtered_dlm = signal.filtfilt(B,A,dlm_dt)
    index = find_zero_crossing(filtered_dlm,int(1.0798/muscle_model.sim_dt),1500)
    if(np.shape(index)[0] == 1):
        axs[1].plot(time_vectors[2], filtered_dlm)
        axs[1].scatter(time_vectors[2][index], filtered_dlm[index])
        plt.show(block = False)
        plt.pause(0.0001)
        print(index)
        return True
    '''
    #reference_stim_dig = np.copy(reference_force_2['stimDig'])
    #reference_stim_dig[reference_stim_dig == 0] = np.nan
    #plt.scatter(time_vectors[2],reference_stim_dig * 0,2,'black',alpha = 1)
    #plt.grid(color='grey', linestyle='-', linewidth=0.3)
    #plt.xlabel('Time [s]')
    #plt.ylabel('Force [N]')
    #plt.draw()
    #plt.pause(0.001)



def test_objective(x,muscle_model,time_vector, active_force,reference_force,twitch_data):
    for i, (time,input,reference) in enumerate(zip(time_vector,active_force,reference_force)):
        error = 0
        if( i == 0 or i == 1):
            continue
        muscle_model.set_parameters(x)
        X0 = muscle_model.get_X0()
        muscle_data, [lm, dlm_dt, Lambda,ls] = muscle_model.muscle_response(X0,time,input)
        
        #B, A = signal.butter(2, 0.012, output='ba')
        #filtered_reference = signal.filtfilt(B,A,reference)
        #reference_force_derative = np.gradient(filtered_reference,muscle_model.sim_dt)
        #estimated_force_derative = np.gradient(muscle_data,muscle_model.sim_dt)

        B, A = signal.butter(2, 0.003, output='ba')
        filtered_dlm = signal.filtfilt(B,A,dlm_dt)
        index = find_zero_crossing(filtered_dlm,int(1.0798/muscle_model.sim_dt),1500)

        error += 1/len(muscle_data) * np.sum((muscle_data-reference)**2)
        if(np.shape(index)[0] == 0):
            error += 100/len(muscle_data) * np.sum((muscle_data-reference)**2)
            error += 500
        elif(np.shape(index)[0] > 1):
            error += np.shape(index)[0] * 2/len(muscle_data) * np.sum((muscle_data-reference)**2)
            error += 500
        if(np.shape(index)[0] == 1):
            error += 1/len(muscle_data) * np.sum((muscle_data-reference)**2)
            print(index)
            result,[left,right] = check_signum(filtered_dlm,index[0],int(0.08/muscle_model.sim_dt))
            error += 10 * np.sum([left,right])
    return error



def objective(x,muscle_model,time_vector, active_force,reference_force):
    '''
    muscle_model.set_parameters(x)
    X0 = muscle_model.get_X0()
    error = 0
    #B, A = signal.butter(2, 0.012, output='ba')
    #B, A = signal.butter(3, 0.003, output='ba')

    muscle_data,_ = muscle_model.muscle_response(X0,time_vector,active_force)
    #filtered_reference = signal.filtfilt(B,A, reference_force)
    #filtered_muscle_data = signal.filtfilt(B,A, muscle_data)
    error += 1/len(muscle_data) * np.sum((muscle_data[0:int(0.7/muscle_model.sim_dt)] - reference_force[0:int(0.7/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
    error += 10/len(muscle_data) * np.sum((muscle_data[int(0.7/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)] - reference_force[int(0.7/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
    error += 1/len(muscle_data) * np.sum((muscle_data[int(1.395/muscle_model.sim_dt)::] - reference_force[int(1.395/muscle_model.sim_dt)::])**2)*muscle_model.sim_dt
    return error
    '''
    error = 0
    
    
    '''
    #--------------
    twitch_duration, twitch_amplitude = twitch_data
    twitch_duration = x[14]
    twitch_amplitude = x[15]
    x = x[0:14]

    low_frequency = 35
    high_frequency = 75
    twitch_delay = 0.0025
    low_freq_duration = int((19 * ((1/low_frequency - muscle_model.sim_dt/1.5)))/muscle_model.sim_dt)
    
    high_freq_duration = int((23 * ((1/high_frequency + muscle_model.sim_dt/1.5)))/muscle_model.sim_dt)
    input_1_low_freq = muscle_active_force.parabolic_twitch(time_vector[0:low_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, low_frequency, muscle_model.sim_dt)
    input_1_high_freq = muscle_active_force.parabolic_twitch(time_vector[low_freq_duration:low_freq_duration+high_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, high_frequency, muscle_model.sim_dt)
    input_1_low_freq_end = muscle_active_force.parabolic_twitch(time_vector[low_freq_duration+high_freq_duration:2*low_freq_duration+high_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, low_frequency, muscle_model.sim_dt)
    zeros = np.zeros([len(time_vector)-(2*low_freq_duration+high_freq_duration)])
    active_force = np.concatenate((input_1_low_freq, input_1_high_freq,input_1_low_freq_end,zeros), axis=0)
    #--------------
    '''

    muscle_model.set_parameters(x)
    X0 = muscle_model.get_X0()
    #muscle_data,_ = muscle_model.muscle_response(X0,time_vector,active_force)

    #plt.plot(time_vector,estimated_force_derative)
    #plt.plot(time_vector,reference_force_derative)
    #plt.show()
    #exit()
    
    '''
    plt.plot(time_vector[0:int(0.95/muscle_model.sim_dt)], muscle_data[0:int(0.95/muscle_model.sim_dt)])
    plt.plot(time_vector[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)], muscle_data[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])
    plt.plot(time_vector[int(1.395/muscle_model.sim_dt)::], muscle_data[int(1.395/muscle_model.sim_dt)::])

    plt.plot(time_vector[0:int(0.95/muscle_model.sim_dt)], reference_force[0:int(0.95/muscle_model.sim_dt)])
    plt.plot(time_vector[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)], reference_force[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])
    plt.plot(time_vector[int(1.395/muscle_model.sim_dt)::], reference_force[int(1.395/muscle_model.sim_dt)::])
    plt.show()
    '''
    '''
    error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative-reference_force_derative)**2)
    error += 1/len(muscle_data) * np.sum((muscle_data[0:int(0.16/muscle_model.sim_dt)] - reference_force[0:int(0.16/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
    error += 1/len(muscle_data) * np.sum((muscle_data[int(0.16/muscle_model.sim_dt):int(0.95/muscle_model.sim_dt)] - reference_force[int(0.16/muscle_model.sim_dt):int(0.95/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
    error += 1/len(muscle_data) * np.sum((muscle_data[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)] - reference_force[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
    error += 1/len(muscle_data) * np.sum((muscle_data[int(1.395/muscle_model.sim_dt)::] - reference_force[int(1.395/muscle_model.sim_dt)::])**2)*muscle_model.sim_dt
    return error
    '''


    muscle_data,_ = muscle_model.muscle_response(X0,time_vector,active_force)
    B, A = signal.butter(2, 0.012, output='ba')
    filtered_reference = signal.filtfilt(B,A,reference_force)
    filtered_estimated_force= signal.filtfilt(B,A,muscle_data)
    reference_force_derative = np.gradient(filtered_reference,muscle_model.sim_dt)
    estimated_force_derative = np.gradient(filtered_estimated_force,muscle_model.sim_dt)

    error += 1/len(muscle_data) * np.sum((muscle_data[0:int(0.16/muscle_model.sim_dt)] - reference_force[0:int(0.16/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
    error += 10/len(muscle_data) * np.sum((muscle_data[int(0.16/muscle_model.sim_dt):int(0.95/muscle_model.sim_dt)] - reference_force[int(0.16/muscle_model.sim_dt):int(0.95/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
    error += 10/len(muscle_data) * np.sum((muscle_data[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)] - reference_force[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
    error += 1/len(muscle_data) * np.sum((muscle_data[int(1.395/muscle_model.sim_dt)::] - reference_force[int(1.395/muscle_model.sim_dt)::])**2)*muscle_model.sim_dt

    return error

    for i, (time,input,reference) in enumerate(zip(time_vector,active_force,reference_force)):
        '''
        if( i == 0):
            input = muscle_active_force.parabolic_twitch(time,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
        if( i == 1):
            input = muscle_active_force.parabolic_twitch(time,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
            input[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal
        '''
        #muscle_model.set_parameters(x)
        #X0 = muscle_model.get_X0()
        muscle_data,_ = muscle_model.muscle_response(X0,time,input)
        B, A = signal.butter(2, 0.012, output='ba')
        filtered_reference = signal.filtfilt(B,A,reference)
        filtered_estimated_force= signal.filtfilt(B,A,muscle_data)
        reference_force_derative = np.gradient(filtered_reference,muscle_model.sim_dt)
        estimated_force_derative = np.gradient(filtered_estimated_force,muscle_model.sim_dt)
        
        '''
        plt.plot(time[0:int(0.95/muscle_model.sim_dt)], muscle_data[0:int(0.95/muscle_model.sim_dt)])
        plt.plot(time[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)], muscle_data[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])
        plt.plot(time[int(1.395/muscle_model.sim_dt)::], muscle_data[int(1.395/muscle_model.sim_dt)::])

        plt.plot(time[0:int(0.95/muscle_model.sim_dt)], reference[0:int(0.95/muscle_model.sim_dt)])
        plt.plot(time[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)], reference[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])
        plt.plot(time[int(1.395/muscle_model.sim_dt)::], reference[int(1.395/muscle_model.sim_dt)::])
        plt.show()
        '''

        #error += 1/len(muscle_data) * np.sum((muscle_data[0:int(0.95/muscle_model.sim_dt)] - reference[0:int(0.95/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
        #error += 10/len(muscle_data) * np.sum((muscle_data[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)] - reference[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
        #error += 1/len(muscle_data) * np.sum((muscle_data[int(1.395/muscle_model.sim_dt)::] - reference[int(1.395/muscle_model.sim_dt)::])**2)*muscle_model.sim_dt

        #B, A = signal.butter(2, 0.012, output='ba')
        #filtered_reference = signal.filtfilt(B,A,reference)
        #reference_force_derative = np.gradient(filtered_reference,muscle_model.sim_dt)
        #estimated_force_derative = np.gradient(muscle_data,muscle_model.sim_dt)
        
        if (i == 0):
            continue
            error += 3/len(muscle_data) * np.sum((muscle_data-reference)**2)
            error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative-reference_force_derative)**2)
            #plt.plot(time,np.gradient(signal.filtfilt(B,A,reference)))
            #plt.plot(time,np.gradient(muscle_data))
            #plt.plot(time,signal.filtfilt(B,A,reference))
            #plt.plot(time,reference)
            #plt.show()
            #exit()
        if(i == 1):
            continue
            error += 1/len(muscle_data) * np.sum((muscle_data - reference)**2)
            #error += 1/len(muscle_data) * np.sum((muscle_data[0:int(0.25/muscle_model.sim_dt)] - filtered_reference[0:int(0.25/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
            #error += 1/len(muscle_data) * np.sum((muscle_data[int(0.2/muscle_model.sim_dt)::] - filtered_reference[int(0.2/muscle_model.sim_dt)::])**2)*muscle_model.sim_dt
            #plt.plot(filtered_reference[0:int(0.25/muscle_model.sim_dt)])
            #plt.plot(filtered_reference[int(0.25/muscle_model.sim_dt)::])
            #plt.show()
            #error += 1/len(muscle_data) * np.sum((muscle_data - reference)**2)
            error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative-reference_force_derative)**2)
            #error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative[0:int(0.2/muscle_model.sim_dt)] - reference_force_derative[0:int(0.2/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
            #error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative[int(0.2/muscle_model.sim_dt)::] - reference_force_derative[int(0.2/muscle_model.sim_dt)::])**2)*muscle_model.sim_dt
            #plt.plot(filtered_reference[0:int(0.2/muscle_model.sim_dt)])
            #plt.show()
            '''
            plt.plot(time,estimated_force_derative)
            plt.plot(time,reference_force_derative)
            plt.figure()
            
            plt.plot(time,muscle_data)
            plt.plot(time,reference)
            
            plt.show()
            exit()
            '''
            #error += 1/len(muscle_data) * np.sum((muscle_data-reference)**2)
            #error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative-reference_force_derative)**2)
            #estimated_force_derative = np.gradient(muscle_data, muscle_model.sim_dt)
            #plt.plot(time,estimated_force_derative)
            #plt.figure()
            #plt.plot(time,muscle_data)
            #plt.show()
        if(i == 2):
            #error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative-reference_force_derative)**2)
            error += 1/len(muscle_data) * np.sum((muscle_data[0:int(0.16/muscle_model.sim_dt)] - reference[0:int(0.16/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
            error += 1/len(muscle_data) * np.sum((muscle_data[int(0.16/muscle_model.sim_dt):int(0.95/muscle_model.sim_dt)] - reference[int(0.16/muscle_model.sim_dt):int(0.95/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
            error += 10/len(muscle_data) * np.sum((muscle_data[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)] - reference[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
            error += 1/len(muscle_data) * np.sum((muscle_data[int(1.395/muscle_model.sim_dt)::] - reference[int(1.395/muscle_model.sim_dt)::])**2)*muscle_model.sim_dt
            #muscle_data,_ = muscle_model.muscle_response(X0,time,input)
            #filtered_reference = signal.filtfilt(B,A, reference)
            #filtered_muscle_data = signal.filtfilt(B,A, muscle_data)
            #plt.plot(filtered_muscle_data)
            #plt.plot(filtered_reference)
            #plt.plot(reference)
            #plt.show()
            #reference_force_derative = np.gradient(filtered_reference, muscle_model.sim_dt)
            #estimated_force_derative = np.gradient(muscle_data, muscle_model.sim_dt)
            #plt.plot(estimated_force_derative)
            #plt.show()
            #plt.plot(time,muscle_data)
            #plt.plot(time,filtered_reference)
            #print(x)
            #plt.show()
            #error += 1/len(muscle_data) * np.sum((muscle_data-reference)**2)

            #error += np.sum((muscle_data-reference)**2)
            '''
            if(i==2):
                #error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative[0:int(0.5/muscle_model.sim_dt)] - reference_force_derative[0:int(0.5/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
                #error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative[int(0.5/muscle_model.sim_dt):int(1.0/muscle_model.sim_dt)] - reference_force_derative[int(0.5/muscle_model.sim_dt):int(1.0/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
                #error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - reference_force_derative[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
                #error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative[int(1.25/muscle_model.sim_dt)::] - reference_force_derative[int(1.25/muscle_model.sim_dt)::])**2)*muscle_model.sim_dt

                #error += 1/len(muscle_data) * np.sum((muscle_data[0:int(0.5/muscle_model.sim_dt)] - filtered_reference[0:int(0.5/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
                #error += 1/len(muscle_data) * np.sum((muscle_data[int(0.5/muscle_model.sim_dt):int(1.0/muscle_model.sim_dt)] - filtered_reference[int(0.5/muscle_model.sim_dt):int(1.0/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
                #error += 10/len(muscle_data) * np.sum((muscle_data[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)] - filtered_reference[int(1.0/muscle_model.sim_dt):int(1.25/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
                #error += 1/len(muscle_data) * np.sum((muscle_data[int(1.25/muscle_model.sim_dt)::] - filtered_reference[int(1.25/muscle_model.sim_dt)::])**2)*muscle_model.sim_dt


                error += 1/len(muscle_data) * np.sum((muscle_data[0:int(0.7/muscle_model.sim_dt)] - reference[0:int(0.7/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
                error += 10/len(muscle_data) * np.sum((muscle_data[int(0.7/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)] - reference[int(0.7/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
                error += 1/len(muscle_data) * np.sum((muscle_data[int(1.395/muscle_model.sim_dt)::] - reference[int(1.395/muscle_model.sim_dt)::])**2)*muscle_model.sim_dt
                error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative-reference_force_derative)**2)*muscle_model.sim_dt
                #error += 1/len(muscle_data) * np.sum((muscle_data-filtered_reference)**2)
            '''
            if(i==1):
                continue
                #print(muscle_data[0:0.3/0.0001])
                #filtered_muscle_force = medfilt(muscle_data,99)
                #filtered_reference = medfilt(reference,99)
                #error += 1/len(muscle_data) * np.sum((estimated_force_derative-reference_force_derative)**2) * muscle_model.sim_dt
                #error += 1/len(muscle_data) * np.sum((muscle_data-reference)**2)


                #error += 0
                #error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative[0:int(0.15/muscle_model.sim_dt)] - reference_force_derative[0:int(0.15/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
                #error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative[int(0.15/muscle_model.sim_dt):int(0.45/muscle_model.sim_dt)] - reference_force_derative[int(0.15/muscle_model.sim_dt):int(0.45/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
                #error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative[int(0.45/muscle_model.sim_dt):int(0.53/muscle_model.sim_dt)] - reference_force_derative[int(0.45/muscle_model.sim_dt):int(0.53/muscle_model.sim_dt)])**2)*muscle_model.sim_dt
                #error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative[int(0.53/muscle_model.sim_dt)::] - reference_force_derative[int(0.53/muscle_model.sim_dt)::])**2)*muscle_model.sim_dt
                '''
                error += 0
                error += 1/len(muscle_data) * np.sum((muscle_data[0:int(0.15/muscle_model.sim_dt)] - filtered_reference[0:int(0.15/muscle_model.sim_dt)])**2)
                error += 1/len(muscle_data) * np.sum((muscle_data[int(0.15/muscle_model.sim_dt):int(0.45/muscle_model.sim_dt)] - filtered_reference[int(0.15/muscle_model.sim_dt):int(0.45/muscle_model.sim_dt)])**2)
                error += 1/len(muscle_data) * np.sum((muscle_data[int(0.45/muscle_model.sim_dt):int(0.53/muscle_model.sim_dt)] - filtered_reference[int(0.45/muscle_model.sim_dt):int(0.53/muscle_model.sim_dt)])**2)
                error += 1/len(muscle_data) * np.sum((muscle_data[int(0.53/muscle_model.sim_dt)::] - filtered_reference[int(0.53/muscle_model.sim_dt)::])**2)
                '''
                error += 1/len(muscle_data) * np.sum((muscle_data-reference)**2)

            #error += 1/len(muscle_data) * np.sum((muscle_data-reference)**2)
            #error += np.sum((muscle_data-reference)**2)
            if(i==0):
                continue
                #sig_fft = fftpack.fft(sig)

                #error += 0
                #error += 1/len(estimated_force_derative) * np.sum((estimated_force_derative-reference_force_derative)**2)*muscle_model.sim_dt
                error += 1/len(muscle_data) * np.sum((muscle_data-reference)**2)
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

def muscle_identification(muscle_model,time_vector,active_force,reference_force,twitch_data,damping_ratio,damping_ration_margin,bounds,threads,disp_debug):
    x0 = muscle_model.get_parameters()
    

    
    #x0.append(twitch_data[0]) # duration
    #x0.append(twitch_data[1]) # amplitude

    #drc = NonlinearConstraint(muscle_model.get_damping_ratio, damping_ratio - damping_ration_margin, damping_ratio + damping_ration_margin)
    #src = NonlinearConstraint(muscle_model.get_stiffness_ratio, 10, 10000)
    result = differential_evolution(objective,x0 = x0,args = (muscle_model, time_vector[2], active_force[2], reference_force[2]),constraints=(),bounds=bounds,workers = threads, disp = disp_debug,polish=False,callback=print_state)
    #result = differential_evolution(objective,x0 = x0,args = (muscle_model, time_vector, active_force, reference_force,twitch_data),constraints=(),bounds=bounds,workers = threads, disp = disp_debug,polish=False,callback=print_state)
    #result = differential_evolution(objective,x0 = x0,args = (muscle_model, time_vector, active_force, reference_force),constraints=(),bounds=bounds,workers = threads, disp = disp_debug,polish=False)
    return result

def muscle_grid_search(muscle_model,time_vector,active_force,reference_force,bounds,threads,disp_debug):
    brute_range = tuple()
    for bound in bounds:
        step = np.abs(bound[1] - bound[0])/4 # 10 different values to test
        
        brute_range = brute_range + (slice(bound[0],bound[1],step),)
    result = brute(objective, brute_range, args=(muscle_model, time_vector, active_force, reference_force),workers = threads, disp=disp_debug)
    return result

def muscle_optimization():
    # muscle parameters
    sim_dt = 0.0001
    twitch_duration = 0.01
    twitch_delay = 0.0025
    twitch_frequency = 40
    twitch_amplitude = 1
    #km = 0.1
    #kt = 1
    #m = 0.002
    #c = 1 * 2 * m * np.sqrt((kt+km)/m)

    km = 4.73751960e+00
    kt = 7.44633205e+01
    m = 3.16944811e-02
    c = 3.11960593e+00
    delta = 0.01
    F_K = 1
    F_delta = 1
    # identification parameters
    disp_debug = True
    twitch_duration_bound = (0.001,0.025)
    twitch_delay_bound = (0,0.005)
    twitch_amplitude_bound = (0.01,100)
    km_bound = (0.0001,1200)
    kt_bound = (0.0001,1200)
    m_bound = (0.0001,0.0799)
    c_bound = (0.0001,1200)
    FK_bound = (1,1200)
    Fdelta_bound = (0.001,1000)
    delta_bound = (0.001,1)
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
    filenames = ['947kLHLf/947kLHLf_trial_02','947kLHLf/947kLHLf_trial_07']
    # take both signals separately during optimization
    #reference_forces = prepare_reference_data(filenames,sim_dt)
    reference_force_0 = prepare_reference_data_in(filenames[0],sim_dt,steady_time=0.075,zeros_time=3.5)
    reference_force_1 = prepare_reference_data_in(filenames[1],sim_dt,steady_time=0.6,zeros_time=0.3)
    reference_forces = [reference_force_0['force'],reference_force_1['force']]
    time_vector_0 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    input_0 = muscle_active_force.parabolic_twitch(time_vector_0,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    input_1 = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    input_1[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal
    time_vectors = [time_vector_0,time_vector_1]
    active_forces = [input_0,input_1]

    muscle_force,[lm,dlm_dt] = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1])
    #plt.figure(1)
    #plt.plot(time_vectors[1],muscle_force,'tab:blue')
    #plt.plot(time_vectors[1],reference_forces[1], 'tab:orange')

    #plt.xlabel('Time [s]')
    #plt.ylabel('Force [N]')
    #plt.show()
    #exit()
    twitch_data = (twitch_duration,twitch_delay, twitch_amplitude, twitch_frequency,sim_dt)
    result = muscle_identification(muscle_model,time_vectors,active_forces,reference_forces,twitch_data,damping_ratio,damping_ratio_margin,bounds,threads,disp_debug)
    print(result.x)

    '''
    twitch_duration = result.x[4]
    twitch_delay = result.x[5]
    twitch_amplitude = result.x[6]
    result.x = result.x[0:4]
    '''

    input_0 = muscle_active_force.parabolic_twitch(time_vector_0,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    input_1 = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    input_1[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal

    muscle_model.set_parameters(result.x)
    X0 = muscle_model.get_X0()
    fitted_muscle_data_0,_ = muscle_model.muscle_response(X0,time_vectors[0],input_0)
    fitted_muscle_data_1,_ = muscle_model.muscle_response(X0,time_vectors[1],input_1)

    B, A = signal.butter(2, 0.012, output='ba')
    plt.figure(0)
    plt.plot(time_vectors[0], reference_forces[0])
    plt.plot(time_vectors[0], signal.filtfilt(B,A, reference_forces[0]))
    plt.plot(time_vectors[0], fitted_muscle_data_0)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.figure(1)
    plt.plot(time_vectors[1], reference_forces[1])
    plt.plot(time_vectors[1], signal.filtfilt(B,A, reference_forces[1]))
    plt.plot(time_vectors[1], fitted_muscle_data_1)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')

    reference_stim_dig = np.copy(reference_force_1['stimDig'])
    reference_stim_dig[reference_stim_dig == 0] = np.nan
    plt.scatter(time_vector_1,reference_stim_dig * 0,2,'black',alpha = 1)
    plt.grid(color='grey', linestyle='-', linewidth=0.3)
    plt.show()


def rheopectic_muscle_optimization():
    sim_dt = 0.0001
    twitch_duration = 0.01
    twitch_delay = 0.0025
    #input 1a
    twitch_frequency = 40
    #input 1
    low_frequency = 35
    high_frequency = 75
    #...
    twitch_amplitude = 1
    
    muscle_model = create_rheopectic_model(sim_dt)

    # identification parameters
    disp_debug = True
    k1_bound = (0.01,5000)
    k2_bound = (0.01,5000)
    C_bound = (2,5)
    D_bound = (1,5)
    c1_bound = (0.1,100)
    c2_bound = (0.1,100)
    lambda0_bound = (0.1,1)
    F0_bound = (-0.5,0.5)
    c_rh_min_bound = (muscle_model.c_rh_min/50,muscle_model.c_rh_min*50)
    
    c_rh_bound = (muscle_model.c_rh/100,muscle_model.c_rh*100)
    c1_bound = (muscle_model.c1/100,muscle_model.c1*100)
    #([  0.28579205, 109.77117007,   1.11604163,  19.96163719,
    #     1.44207403,   0.21671702])
    bounds = (c_rh_bound,c_rh_min_bound,c1_bound,c2_bound,k1_bound,k2_bound,C_bound,D_bound,lambda0_bound,F0_bound)
    threads = 1
    damping_ratio = 1.
    damping_ratio_margin = 0.1

    # create hill model
    
    #exit()
    X0 = muscle_model.get_X0()
    #exit()
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
    X0 = muscle_model.get_X0()


    #reference_force_0 = prepare_reference_data_in(filenames[0],sim_dt,steady_time=1.5,zeros_time=0.3)
    #reference_force_1 = prepare_reference_data_in(filenames[1],sim_dt,steady_time=0.6,zeros_time=0.3)
    #reference_forces = [reference_force_0['force'],reference_force_1['force']]
    #time_vector_0 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    #input_0 = muscle_active_force.parabolic_twitch(time_vector_0,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    #time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    #input_1 = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    #input_1[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal
    time_vectors = [time_vector_00,time_vector_0,time_vector_1]
    active_forces = [input_0,input_1a,input_1]

    reference_stim_dig = np.copy(reference_force_1['stimDig'])
    reference_stim_dig[reference_stim_dig == 0] = np.nan

    # take both signals separately during optimization
    #reference_forces = prepare_reference_data(filenames,sim_dt)
    #time_vector_0 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    #input_0 = muscle_active_force.parabolic_twitch(time_vector_0,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    #time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    #input_1 = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    #input_1[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal
    #time_vectors = [time_vector_0,time_vector_1]

    #active_forces = [input_0,input_1]

    B, A = signal.butter(2, 0.003, output='ba')
    muscle_force,[lm,dlm_dt,ls] = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1])
    #plt.plot(time_vectors[1],signal.filtfilt(B,A,reference_forces[1]), 'tab:orange')
    #B, A = signal.butter(2, 0.003, output='ba')
    plt.plot(time_vectors[1],muscle_force)
    plt.plot(time_vectors[1],signal.filtfilt(B,A,reference_forces[1]), 'tab:blue')
    plt.grid()
    plt.figure()
    plt.plot(time_vectors[1],np.gradient(signal.filtfilt(B,A,reference_forces[1]),sim_dt), 'tab:orange')
    plt.grid()
    plt.show()
    #exit()

    twitch_data = (twitch_duration,twitch_delay, twitch_amplitude, twitch_frequency,sim_dt)
    result = muscle_identification(muscle_model,time_vectors,active_forces,reference_forces,twitch_data,damping_ratio,damping_ratio_margin,bounds,threads,disp_debug)
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
    km = 4.73751960e+00
    kt = 7.44633205e+01
    m = 3.16944811e-02
    c = 3.11960593e+00
    #c = 1 * 2 * m * np.sqrt((kt+km)/m)
    delta = 0.01
    F_K = 100
    F_delta = 10
    active_force = muscle_active_force.parabolic_twitch(time_vector,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    active_force[int(0.525/sim_dt)::]=0
    muscle_model = hill_muscle_model(km,kt,m,c,delta,sim_dt)
    #muscle_model = hill_muscle_model(km,kt,m,c,delta,sim_dt)
    X0 = muscle_model.get_X0()
    muscle_force,[lm,dlm_dt] = muscle_model.muscle_response(X0,time_vector,active_force)
    plt.figure(1)
    plt.plot(time_vector,muscle_force)
    stim_dig = np.copy(active_force)
    stim_dig[stim_dig == 0] = np.nan
    plt.scatter(time_vector,stim_dig * 0,2,'black',alpha = 1)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.grid(color='grey', linestyle='-', linewidth=0.3)
    #plt.figure(2)
    #plt.plot(time_vector,dlm_dt)
    #print('Median dlm_dt', np.median(dlm_dt))
    #y_bottom,y_top= plt.gca().get_ylim()
    #plt.scatter(time_vector,stim_dig * 0 - np.abs(y_bottom),2,'black',alpha = 1)
    #plt.xlabel('Time [s]')
    #plt.ylabel('dlm_dt')
    #plt.grid(color='grey', linestyle='-', linewidth=0.3)
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

    km = 1.16576887e+00
    kt = 6.35540408e+01
    m = 2.90577854e-02
    c = 3.01295504e+00
    delta = 0.01
    A = 1
    B = 1

    #min_c = 1.03821004e+02
    #max_c = 3.76895335e-01
    max_viscosity = 7.47669011e+01
    min_viscosity = 1.02635792e-01
    k1 = 1.11900304e+00
    k2 = 5.18992254e+00
    C = 4.99977965e+00
    D = 1
    c0 = 25
    lambda0 = 0.05
    
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


    twitch_frequency
    input_test = muscle_active_force.parabolic_twitch(time_vector,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    input_test[int(0.925/sim_dt)::]=0 # to be the same like the orginal signal
    muscle_force,[lm,dlm_dt,Lambda] = muscle_model.muscle_response(X0,time_vector,input_test)
    viscosity_from_lambda = c0 * Lambda
    plt.figure(1)
    plt.plot(time_vector,dlm_dt)
    #np.savetxt('dlm_dt.csv', (dlm_dt), delimiter=',')
    plt.figure(2)
    plt.plot(time_vector,Lambda)
    plt.show()
    exit()

    input_1_low_freq_end = muscle_active_force.parabolic_twitch(time_vector[low_freq_duration+high_freq_duration:2*low_freq_duration+high_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, low_frequency, sim_dt)

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

def modified_muscle_simulation():
    sim_dt = 0.0001
    sim_time = 0.8
    time_vector = np.arange(0,sim_time,sim_dt)
    twitch_duration = 0.01
    twitch_delay = 0.0025
    twitch_frequency = 40
    twitch_amplitude = 1
    km = 4.73751960e+00
    kt = 7.44633205e+01
    m = 3.16944811e-02
    c = 3.11960593e+00
    c1 = 1.08358338e+00 
    cs = 0.2
    ks = 0.1
    #c = 1 * 2 * m * np.sqrt((kt+km)/m)
    delta = 0.01
    F_K = 100
    F_delta = 10
    ls0 = -0.001
    F0 = 0.7
    active_force = muscle_active_force.parabolic_twitch(time_vector,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    active_force[int(0.525/sim_dt)::]=0
    muscle_model = modified_hill_muscle_model(km,kt,m,c,c1,cs,ks,ls0,F0,delta,sim_dt)
    #muscle_model = hill_muscle_model(km,kt,m,c,delta,sim_dt)
    X0 = muscle_model.get_X0()
    muscle_force,[lm,dlm_dt,ls] = muscle_model.muscle_response(X0,time_vector,active_force)
    plt.figure(1)
    #plt.plot(time_vector,lm)
    #plt.plot(time_vector,ls)
    plt.plot(time_vector,muscle_force)
    #plt.plot(muscle_model.test)
    stim_dig = np.copy(active_force)
    stim_dig[stim_dig == 0] = np.nan
    plt.scatter(time_vector,stim_dig * 0,2,'black',alpha = 1)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.grid(color='grey', linestyle='-', linewidth=0.3)
    #plt.figure(2)
    #plt.plot(time_vector,dlm_dt)
    #print('Median dlm_dt', np.median(dlm_dt))
    #y_bottom,y_top= plt.gca().get_ylim()
    #plt.scatter(time_vector,stim_dig * 0 - np.abs(y_bottom),2,'black',alpha = 1)
    #plt.xlabel('Time [s]')
    #plt.ylabel('dlm_dt')
    #plt.grid(color='grey', linestyle='-', linewidth=0.3)
    plt.show()

def modified_muscle_optimization():
    # muscle parameters
    sim_dt = 0.0001
    twitch_duration = 0.01
    twitch_delay = 0.0025
    twitch_frequency = 40
    twitch_amplitude = 1
    km = 8.21688727e+00
    kt = 5.45036325e+01
    m = 2.86717809e-02
    c = 3.22254959e+00
    c1 = 2.03688796e-01 
    cs = 8.21556777e+02
    ks = 8.70843312e+02
    delta = -0.01
    ls0 = -0.5
    F_K = 1
    F_delta = 1
    F0 = 0
    # identification parameters
    disp_debug = True
    km_bound = (0.001,1200)
    kt_bound = (0.001,1200)
    m_bound = (0.0001,0.0799)
    c_bound = (0.001,1200)
    cs_bound = (0.001,1200)
    c1_bound = (0.001,2000)
    ks_bound = (0.001,1200)
    ls0_bound = (-0.5,0.5)
    FK_bound = (1,1200)
    Fdelta_bound = (0.001,1000)
    F0_bound = (-1,1)
    twitch_duration_bound = (0.001,0.025)
    twitch_amplitude_bound = (0.01,100)
    twitch_delay_bound = (0,0.01)
    #bounds = (km_bound,kt_bound,m_bound,c_bound)
    #bounds = (km_bound,kt_bound,m_bound,c_bound,c1_bound,cs_bound,ks_bound,ls0_bound,F0_bound)
    bounds = (c_bound,c1_bound,cs_bound,ks_bound,ls0_bound,F0_bound)
    threads = 12
    damping_ratio = 1.
    damping_ratio_margin = 0.1

    # create hill model
    muscle_model = modified_hill_muscle_model(km,kt,m,c,c1,cs,ks,ls0,F0,delta,sim_dt)
    #muscle_model = hill_muscle_model(km,kt,m,c,delta,sim_dt)
    X0 = muscle_model.get_X0()
    # load reference data
    filenames = ['947kLHLf/947kLHLf_trial_04','947kLHLf/947kLHLf_trial_07']
    # take both signals separately during optimization
    #reference_forces = prepare_reference_data(filenames,sim_dt)
    reference_force_0 = prepare_reference_data_in(filenames[0],sim_dt,steady_time=0.075,zeros_time=3.5)
    reference_force_1 = prepare_reference_data_in(filenames[1],sim_dt,steady_time=1.9,zeros_time=0.3)
    reference_forces = [reference_force_0['force'],reference_force_1['force']]
    time_vector_0 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    input_0 = muscle_active_force.parabolic_twitch(time_vector_0,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    input_1 = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    input_1[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal
    time_vectors = [time_vector_0,time_vector_1]
    active_forces = [input_0,input_1]



    #plt.figure(1)
    B, A = signal.butter(2, 0.003, output='ba')
    muscle_force,[lm,dlm_dt,ls] = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1])
    #plt.plot(time_vectors[1],signal.filtfilt(B,A,reference_forces[1]), 'tab:orange')
    #B, A = signal.butter(2, 0.003, output='ba')
    #plt.plot(time_vectors[1],signal.filtfilt(B,A,reference_forces[1]), 'tab:blue')
    #plt.grid()
    #plt.figure()
    #plt.plot(time_vectors[1],np.gradient(signal.filtfilt(B,A,reference_forces[1]),sim_dt), 'tab:orange')
    #plt.grid()
    #plt.plot(time_vectors[1],muscle_force, 'tab:blue')
    #plt.plot(time_vectors[1],reference_forces[1], 'tab:orange')
    #plt.plot(time_vectors[1],reference_forces[1], 'tab:orange')
    #plt.show()
    #exit()

    #plt.xlabel('Time [s]')
    #plt.ylabel('Force [N]')
    #plt.show()
    #exit()
    twitch_data = (twitch_duration,twitch_delay, twitch_amplitude, twitch_frequency,sim_dt)
    result = muscle_identification(muscle_model,time_vectors,active_forces,reference_forces,twitch_data,damping_ratio,damping_ratio_margin,bounds,threads,disp_debug)
    print(result)
    '''
    twitch_duration = result.x[9]
    twitch_delay = result.x[10]
    twitch_amplitude = result.x[11]
    result.x = result.x[0:9]
    '''
    muscle_model.set_parameters(result.x)
    X0 = muscle_model.get_X0()
    active_forces[0] = muscle_active_force.parabolic_twitch(time_vectors[0],twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    active_forces[1] = muscle_active_force.parabolic_twitch(time_vectors[1],twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    active_forces[1][int(0.525/sim_dt)::]=0 # to be the same like the orginal signal
    fitted_muscle_data_0,_ = muscle_model.muscle_response(X0,time_vectors[0],active_forces[0])
    fitted_muscle_data_1,_ = muscle_model.muscle_response(X0,time_vectors[1],active_forces[1])

    B, A = signal.butter(2, 0.012, output='ba')
    plt.figure(0)
    plt.plot(time_vectors[0], reference_forces[0])
    plt.plot(time_vectors[0], signal.filtfilt(B,A, reference_forces[0]))
    plt.plot(time_vectors[0], fitted_muscle_data_0)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.figure(1)
    plt.plot(time_vectors[1], reference_forces[1])
    plt.plot(time_vectors[1], signal.filtfilt(B,A, reference_forces[1]))
    plt.plot(time_vectors[1], fitted_muscle_data_1)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')

    reference_stim_dig = np.copy(reference_force_1['stimDig'])
    reference_stim_dig[reference_stim_dig == 0] = np.nan
    plt.scatter(time_vector_1,reference_stim_dig * 0,2,'black',alpha = 1)
    plt.grid(color='grey', linestyle='-', linewidth=0.3)
    plt.show()


def rheopectic_modified_muscle_optimization():
    # changed filtering method
    sim_dt = 0.0001
    twitch_duration = 0.01
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


    #km = 2.69442507e+00
    #kt = 6.11709616e+01
    #m = 3.07412836e-02
    #c = 3.08226301e+00

    #km = 7.05745487e-01 * 1
    #kt = 7.06960318e+01 * 1 * 1.25
    #m = 2.28151620e-02 
    #c = 20.65682892e+00 * 6
    #c_rh = c / 3
    #c_rh_min = c_rh / 5
    #c1 = c  / 5
    #cs = 1.35102053e+01 * 2
    #ks = 2.45926152e+02 * 1
    #ls0 = -0.002
    #c = 1 * 2 * m * np.sqrt((kt+km)/m)
    #delta = 0.01
    #F_K = 1
    #F_delta = 1
    #c0 = 25
    #min_c = 0.3#????
    #max_c = 120 # ???
    #k1 = 10.46033652e+00 * 30
    #k2 = 9.31295195e+01 * 1
    #A = 0
    #B = 1
    #C = 3
    #D = 1
    #lambda0 = 6.11381772e-01
    #delta = 0.01
    # identification parameters
    disp_debug = True
    #max_c_bound = (10,150)
    #min_c_bound = (0.1,0.35)
    muscle_model = create_rheopectic_modified_model(sim_dt)

    
    c_rh_bound = (muscle_model.c_rh/10,muscle_model.c_rh*10)
    c1_bound = (muscle_model.c1/10,muscle_model.c1*10)
    k1_bound = (muscle_model.k1/10,muscle_model.k1*10)
    k2_bound = (muscle_model.k2/10,muscle_model.k2*10)
    A_bound = (muscle_model.A/10,muscle_model.A*10)
    B_bound = (muscle_model.B/10,muscle_model.B*10)
    C_bound = (muscle_model.C/10,muscle_model.C*10)
    D_bound = (muscle_model.D/10,muscle_model.D*10)
    #c0_bound = (0.1,100)
    lambda0_bound = (0.1,1)
    F0_bound = (muscle_model.F0/10,muscle_model.F0*10)
    cs_bound = (muscle_model.cs/10,muscle_model.cs*10)
    c1_bound = (muscle_model.c1/10,muscle_model.c1*10)
    ks_bound = (muscle_model.ks/10,muscle_model.ks*10)
    c_rh_min_bound = (muscle_model.c_rh_min/10,muscle_model.c_rh_min*10)

    km_bound = (muscle_model.km/10,muscle_model.km*10)
    kt_bound = (muscle_model.kt/10,muscle_model.kt*10)
    
    #c_rh_max_bound = (muscle_model.c_rh_max/100,muscle_model.c_rh_max*100)
    #c0_bound = (muscle_model.c0/100,muscle_model.c0*100)
    ls0_bound = (muscle_model.ls0/10,muscle_model.ls0*10)
    twitch_duration_bound = (0.001,0.012)
    twitch_amplitude_bound = (0.01,100)
    #km_bound = (0.1,5000)
    #kt_bound = (0.1,5000)
    #([  0.28579205, 109.77117007,   1.11604163,  19.96163719,
    #     1.44207403,   0.21671702])
    #bounds = (min_c_bound,k1_bound,k2_bound, C_bound,D_bound,c0_bound,lambda0_bound,max_c_bound)
    #bounds = (k1_bound,k2_bound, C_bound,D_bound,c0_bound,lambda0_bound,cs_bound,ks_bound,ls0_bound)
    # 1 3 4 6
    #bounds = (k1_bound,k2_bound,c_rh_bound, c_rh_min_bound,ls0_bound,c1_bound,cs_bound,ks_bound,lambda0_bound,A_bound,B_bound,C_bound,D_bound,F0_bound,twitch_duration_bound,twitch_amplitude_bound)
    bounds = (km_bound,kt_bound,c1_bound)
    threads = 12
    damping_ratio = 1.
    damping_ratio_margin = 0.1

    # create hill model
    
    #muscle_model = rheopectic_modified_hill_muscle_model(km,kt,m,cs,ks,ls0,c_rh,c1,c_rh_min,k1,k2,A,B,C,D,lambda0,delta,sim_dt)

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
    X0 = muscle_model.get_X0()


    #reference_force_0 = prepare_reference_data_in(filenames[0],sim_dt,steady_time=1.5,zeros_time=0.3)
    #reference_force_1 = prepare_reference_data_in(filenames[1],sim_dt,steady_time=0.6,zeros_time=0.3)
    #reference_forces = [reference_force_0['force'],reference_force_1['force']]
    #time_vector_0 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    #input_0 = muscle_active_force.parabolic_twitch(time_vector_0,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    #time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    #input_1 = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    #input_1[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal
    time_vectors = [time_vector_00,time_vector_0,time_vector_1]
    active_forces = [input_0,input_1a,input_1]

    reference_stim_dig = np.copy(reference_force_2['stimDig'])
    reference_stim_dig[reference_stim_dig == 0] = np.nan

    # take both signals separately during optimization
    #reference_forces = prepare_reference_data(filenames,sim_dt)
    #time_vector_0 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    #input_0 = muscle_active_force.parabolic_twitch(time_vector_0,twitch_duration,twitch_delay,twitch_amplitude, 1, sim_dt)
    #time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    #input_1 = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    #input_1[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal
    #time_vectors = [time_vector_0,time_vector_1]

    #active_forces = [input_0,input_1]
    #start = time()

    #print(muscle_model.k2)
    #muscle_model.set_parameters([85,muscle_model.k2*100])
    muscle_data,[lm,dlm_dt, Lambda,ls] = muscle_model.muscle_response(X0,time_vectors[2],active_forces[2])
    #muscle_data_1,[lm,dlm_dt, Lambda,ls] = muscle_model.muscle_response(X0,time_vectors[0],active_forces[0])

    B, A = signal.butter(2, 0.012, output='ba')
    filtered_reference = signal.filtfilt(B,A,reference_forces[2])
    filtered_estimated_force= signal.filtfilt(B,A,muscle_data)
    reference_force_derative = np.gradient(filtered_reference,muscle_model.sim_dt)
    estimated_force_derative = np.gradient(filtered_estimated_force,muscle_model.sim_dt)

    #plt.plot(muscle_data)
    ## GRADIENT
    #plt.plot(time_vectors[2],muscle_data)
    #plt.show()
    #exit()
    #stop = time()
    #print(stop-start)
    #exit()
    B, A = signal.butter(2, 0.003, output='ba')
    filtered_reference = signal.filtfilt(B,A,reference_forces[2])
    reference_force_derative = np.gradient(filtered_reference,muscle_model.sim_dt)

    #index, = np.where(np.isclose(reference_force_derative, np.zeros(np.shape(reference_force_derative)),rtol=1e-05, atol=1e-03) == True)

    #index, = np.where((np.diff(reference_force_derative,prepend=0) < 0) == True)
    
    '''
    filtered_dlm = signal.filtfilt(B,A,dlm_dt)
    index = find_zero_crossing(filtered_dlm,int(1.0798/sim_dt),1500)
    result,[left,right] = check_signum(filtered_dlm,int(1.0798/sim_dt),int(0.08/sim_dt))
    
    if(np.shape(index)[0] > 0):
        print('Not detected correct crossing')
    print(index)
    plt.scatter(time_vector_1[index],filtered_dlm[index])
    plt.plot(time_vector_1,filtered_dlm)
    plt.figure()
    plt.plot(time_vector_1,reference_force_derative)
    index = find_zero_crossing(reference_force_derative,int(1.0798/sim_dt),1500)
    plt.scatter(time_vector_1[index],reference_force_derative[index] * 0)
    plt.grid()
    plt.figure()
    #plt.plot(time_vector_1, muscle_data)
    plt.plot(time_vector_1,filtered_reference)
    plt.grid()
    plt.show()
    exit()
    '''
    
    
    plt.plot(time_vectors[2][0:int(0.16/muscle_model.sim_dt)], muscle_data[0:int(0.16/muscle_model.sim_dt)])
    plt.plot(time_vectors[2][int(0.16/muscle_model.sim_dt):int(0.95/muscle_model.sim_dt)], muscle_data[int(0.16/muscle_model.sim_dt):int(0.95/muscle_model.sim_dt)])
    #plt.plot(time_vectors[2][0:int(0.95/muscle_model.sim_dt)], muscle_data[0:int(0.95/muscle_model.sim_dt)])
    plt.plot(time_vectors[2][int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)], muscle_data[int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])
    plt.plot(time_vectors[2][int(1.395/muscle_model.sim_dt)::], muscle_data[int(1.395/muscle_model.sim_dt)::])

    plt.plot(time_vectors[2][0:int(0.95/muscle_model.sim_dt)], reference_forces[2][0:int(0.95/muscle_model.sim_dt)])
    plt.plot(time_vectors[2][int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)], reference_forces[2][int(0.95/muscle_model.sim_dt):int(1.395/muscle_model.sim_dt)])
    plt.plot(time_vectors[2][int(1.395/muscle_model.sim_dt)::], reference_forces[2][int(1.395/muscle_model.sim_dt)::])
    
    plt.figure()
    plt.plot(reference_force_2['stimDig'])
    plt.plot(active_forces[2])
    plt.show()
    exit()
    

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

def create_rheopectic_model(simulation_dt):
    sim_dt = simulation_dt
    km = 4.73751960e+00
    kt = 7.44633205e+01
    m = 3.16944811e-02
    c = 3.11960593e+00
    delta = 0.01
    c_rh = c  * 2
    c_rh_min = (c / 3) / 10
    c1 = c  / 5 
    k1 = 2.79147980e+01
    k2 = 6.98119462e+01
    lambda0 = 6.13209065e-01
    A = 0
    B = 1
    C = 2
    D = 1
    F0 = -4.08750403e-03
    muscle_model = rheopectic_hill_muscle_model(km,kt,m,c_rh,c1,c_rh_min,k1,k2,A,B,C,D,lambda0,F0,delta,sim_dt)
    return muscle_model

def create_rheopectic_modified_model(simulation_dt):
    sim_dt = simulation_dt
    km = 8.21688727e+00
    kt = 5.45036325e+01
    m = 2.86717809e-02
    c = 3.22254959e+00
    #c_rh = c  * 2
    #c_rh_min = (c / 3) / 10
    #c1 = c  / 5 
    cs = 1.35102053e+01 * 2 
    ks = 2.45926152e+02 * 1
    ls0 = -0.0001
    A = 1
    B = 1.5
    C = 3
    D = 0.5
    delta = 0.01

    k1 = 2.79147980e+02
    k2 = 6.98119462e+03
    c_rh = 2.10558506e+01
    c_rh_min = 1.0490643e+00 
    c1 = 4.30490643e+01
    c2 = 0
    #c0 = 9.80490643e-01
    lambda0 = 3.13209065e-01
    F0 = 0

    muscle_model = rheopectic_modified_hill_muscle_model(km,kt,m,cs,ks,ls0,c_rh,c_rh_min,c1,k1,k2,A,B,C,D,lambda0,F0,delta,sim_dt)
    return muscle_model

def rheopectic_modified_muscle_simulation():
    sim_dt = 0.0001
    sim_time = 1.8
    time_vector = np.arange(0,sim_time,sim_dt)
    twitch_duration = 0.01
    twitch_delay = 0.0025
    twitch_amplitude = 1
    twitch_frequency = 40
    low_frequency = 35
    high_frequency = 75


    #km = 7.05745487e-01 * 1
    #kt = 7.06960318e+01 * 1 * 1.25
    #m = 2.28151620e-02 
    #c = 20.65682892e+00 * 6
    #c_rh = c / 3
    #c_rh_min = c_rh / 5
    #c1 = c  / 5
    #cs = 1.35102053e+01 * 2
    #ks = 2.45926152e+02 * 1
    #ls0 = -0.002

    #k1 = 10.46033652e+00 * 30
    #k2 = 9.31295195e+01 * 1
    #A = 0
    #B = 1
    #C = 3
    #D = 1
    #lambda0 = 6.11381772e-01
    #delta = 0.01
    
    #min_c_bound,k1_bound,C_bound,D_bound,c0_bound,lambda0_bound
    # 4.88715531e-01, 2.43961686e+02, 2.00293405e+00, 3.99843875e+01, 4.26249386e-02, 3.23829975e-01
    
    #active_force = muscle_active_force.parabolic_twitch(time_vector,twitch_duration,twitch_delay,twitch_amplitude, 100, sim_dt)
    #active_force[int(0.525/sim_dt)::]=0
    muscle_model = create_rheopectic_modified_model(sim_dt)
    #muscle_model = rheopectic_modified_hill_muscle_model(km,kt,m,cs,ks,ls0,c_rh,c1,c_rh_min,k1,k2,A,B,C,D,lambda0,delta,sim_dt)
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

    input_test = muscle_active_force.parabolic_twitch(time_vector,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    input_test[int(0.925/sim_dt)::]=0 # to be the same like the orginal signal
    start = time()
    muscle_force,[lm, dlm_dt, Lambda,ls] = muscle_model.muscle_response(X0,time_vector,active_force)
    stop = time()
    print('Time ', stop-start)
    #exit()
    #plt.plot(muscle_model.test)
    #plt.show()
    #exit()
    print(np.mean(muscle_model.test))
    B, A = signal.butter(3, 0.003, output='ba')
    #plt.plot(time_vector,dlm_dt)
    #np.savetxt('dlm_dt.csv', (dlm_dt), delimiter=',')
    #plt.plot(time_vector,signal.filtfilt(B,A, dlm_dt))
    fig, axs = plt.subplots(2)
    axs[0].plot(time_vector, muscle_force)
    dlm_dt_temp = dlm_dt.copy()
    dlm_dt_temp[dlm_dt_temp<0] = 0
    #muscle_model.c_rh * Lambda * dlm_dt_temp
    axs[1].plot(time_vector, Lambda)
    axs[0].set_ylabel('Muscle force [N]')
    axs[1].set_ylabel('Lambda [-]')
    axs[1].set_xlabel('Time [s]')
    axs[0].grid()
    axs[1].grid()

    '''
    fig, axs1 = plt.subplots(3)
    axs1[0].plot(time_vector,muscle_model.c_rh * Lambda * dlm_dt_temp)
    axs1[0].set_ylabel('max(0,dlm_dt)*Lambda*c_rh [N]')
    axs1[0].grid()
    axs1[1].plot(time_vector,muscle_model.c_rh_min * dlm_dt)
    axs1[1].set_ylabel('c_rh_min * dlm_dt [N]')
    axs1[1].grid()
    axs1[2].plot(time_vector,muscle_model.c_rh * Lambda * dlm_dt_temp + muscle_model.c_rh_min * dlm_dt)
    axs1[2].set_ylabel('Total rheopectic force [N]')
    axs1[2].grid()
    '''
    #plt.plot(muscle_model.test)
    plt.show()
    exit()
    #viscosity_from_lambda = c0 * Lambda
    #plt.figure(1)
    plt.plot(time_vector,muscle_force)
    #np.savetxt('dlm_dt.csv', (dlm_dt), delimiter=',')
    #plt.figure(2)
    #plt.plot(time_vector,Lambda)
    plt.show()
    #exit()




if __name__=="__main__":
    #muscle_simulation()
    #muscle_optimization()
    #sigmoid_optimization()
    #rheopectic_simulation()
    #rheopectic_muscle_optimization()
    #modified_muscle_simulation()
    #modified_muscle_optimization()
    #rheopectic_modified_muscle_simulation()
    rheopectic_modified_muscle_optimization()
