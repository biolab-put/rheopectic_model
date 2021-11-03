from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import time

## ZMIANA KROKU (Z OBSERWACJI X PODCZAS OPTYMALIZACJI)

delta = 0.03 #30mm - początkowe wydłużenie mięśnia

def objective(x, X0, time_vector, active_force,m,km,kt,k2,A,B,max_c, sim_dt, reference_force):
    min_c = x[0]
    k1 = x[1]
    D = x[2]
    C = x[3]
    c0 = x[4]
    lm0 = kt * delta / (km+kt)
    Y0 = [lm0,0, 0]

    muscle_data =  rheopectic_muscle_response(Y0,time_vector, active_force,m,km,kt,k1,k2, A,B,C,D, max_c, min_c,c0, sim_dt)
    estimated_force = pd.DataFrame(data = {'timestamp': time_vector,'estimated force' : muscle_data['estimated force']})
    return np.sum((estimated_force['estimated force']-reference_force['reference force'])**2)

def load_data(filename):
    data = pd.read_csv(filename)
    return data

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

def solve_muscle_dynamics(t,x,active_force, m, km,kt,k1,k2, A,B,C,D, max_c, min_c,c0, sim_dt):
    lm, dlm, Lambda = x
    Lambda = np.max([0.0, np.min([1.0, Lambda])])
    dLambda = -k1*np.sign(dlm)*(np.abs(dlm)**(A))*(Lambda**(B))+k2*np.sign(dlm)*(np.abs(dlm)**(C))*((1-Lambda)**(D))
    K = 1 - np.sqrt(max_c/min_c)
    c = max_c/((1 - K * Lambda)**2) + c0
    #print(c)
    d2lm = 1/m * (-np.abs(c)*dlm-np.abs(km)*lm+np.abs(kt)*(delta - lm)-active_force[int(t/sim_dt)])
    return [dlm,d2lm,dLambda]


def rheopectic_muscle_response(X0,time_vector, active_force,m,km,kt,k1,k2, A,B,C,D, max_c, min_c,c0, sim_dt):
    solution = solve_ivp(solve_muscle_dynamics, t_span=[0, time_vector[-1]], y0=X0, t_eval = time_vector, args=(active_force, m, km,kt,k1,k2, A,B,C,D, max_c, min_c,c0, sim_dt), method = 'LSODA',first_step = sim_dt, max_step = sim_dt, atol = 1e90, rtol= 1e90)
    estimated_force = np.abs(kt) * (delta - solution.y[0,:])
    return pd.DataFrame(data = {'timestamp': time_vector, 'estimated force': estimated_force, 'lm' : solution.y[0,:], 'dlm': solution.y[1,:]})

def muscle_state():
    sim_dt = 0.001
    sim_time = 5.1
    time_vector = np.arange(0,sim_time,sim_dt)
    twitch_duration = 0.017
    twitch_delay = 0
    twitch_frequency = 4
    twitch_amplitude = 2
    k1 = 5
    k2 = 1
    A = 1
    B = 1
    C = 2
    D = 1
    max_c = 0.05
    min_c = 0.01
    #km = 0.00105502             
    #kt = 0.02917325
    #m = 0.00134382
    km = 0.00328645#, 0.08120704, 0.00391919, 0.03631652         
    kt = 0.08120704
    m = 0.00391919
    c0 = 0 

    # first try
    min_c = 0.79420633
    k1 = 0.49530833
    D = 1.4736452 
    C = 0.5469772
    c0 = -0.18024405

    # second try
    min_c = 0.79420638
    k1 = 0.49530833
    D = 1.47364526
    C = 0.54697723
    c0 = -0.18024402


    
    lm0 = kt * delta / (km+kt)
    X0 = [lm0,0,0] #lm, dlm, lambda

    filenames = ['1006j_trial_00','1006j_trial_00','1006j_trial_00', '1006j_trial_05', '1006j_trial_00']
    reference_data = [load_data(f'./{filename}') for filename in filenames]
    zeros = pd.DataFrame(np.zeros((2000+4000,reference_data[0].shape[1])),columns=reference_data[0].columns)
    reference_data[0]['force'] = reference_data[0]['force'] - reference_data[0]['force'][0]
    reference_data[1]['force'] = reference_data[1]['force'] - reference_data[1]['force'][0]
    reference_data[2]['force'] = reference_data[2]['force'] - reference_data[2]['force'][0]
    reference_data[4]['force'] = reference_data[4]['force'] - reference_data[4]['force'][0]
    reference_data[0] = reference_data[0][0:600]
    reference_data[1] = reference_data[1][0:600]
    reference_data[2] = reference_data[2][0:600]
    reference_data[3] = reference_data[3][0:6000]
    reference_data[4] = reference_data[4][0:600]

    time_vector_twitch_start = np.arange(0,len(reference_data[0]['force']) *sim_dt,sim_dt)
    active_force_twitch_start = parabolic_twitch(time_vector_twitch_start,twitch_duration,0.06,twitch_amplitude, 1, sim_dt)
    
    time_vector_twitches_middle = np.arange(0,len(reference_data[3]['force']) *sim_dt,sim_dt)
    active_force_twitches_middle = parabolic_twitch(time_vector_twitches_middle,twitch_duration,0.06,twitch_amplitude, twitch_frequency, sim_dt)
    # HARDCORE FIX
    active_force_twitches_middle[5200::] = 0

    time_vector_twitch_stop = np.arange(0,len(reference_data[2]['force']) *sim_dt,sim_dt)
    active_force_twitch_stop = parabolic_twitch(time_vector_twitch_stop,twitch_duration,0.06,twitch_amplitude, 1, sim_dt)

    active_force = np.concatenate((active_force_twitch_start,zeros['force'].values,active_force_twitch_start,zeros['force'].values,active_force_twitch_start,zeros['force'].values, active_force_twitches_middle,zeros['force'].values, active_force_twitch_stop, zeros['force'].values))
    #reference_data = pd.concat([reference_data[0],zeros,reference_data[1],zeros,reference_data[2],zeros], ignore_index=True)
    reference_data = pd.concat([reference_data[0],zeros,reference_data[1],zeros,reference_data[2],zeros,reference_data[3],zeros,reference_data[4],zeros], ignore_index=True)
    
    time_vector = np.arange(0,len(reference_data['force'])*sim_dt,sim_dt)

    estimated_muscle_data = rheopectic_muscle_response(X0,time_vector, active_force,m,km,kt,k1,k2, A,B,C,D, max_c, min_c,c0, sim_dt)

    reference_force = pd.DataFrame(data = {'timestamp': time_vector, 'reference force' : reference_data['force']})
    reference_force['reference force'] = reference_force['reference force']/1000

    '''
    reference_data_1 = load_data('1006j_trial_04')
    reference_data_2 = load_data('1006j_trial_01')
    reference_data = []
    filenames = []
    for filename in os.listdir('.'):
        if filename.startswith('1006j_trial_'):
            filenames.append(load_data(filename))

    reference_data = pd.concat(filenames, ignore_index=True)
    reference_force = pd.DataFrame(data = {'reference force' : reference_data['force']})
    plt.plot(reference_data['force'])
    plt.show()
    exit()
    '''

    ax = estimated_muscle_data.plot(x="timestamp", y="estimated force")
    reference_force.plot(x="timestamp", y="reference force",ax=ax)
    #reference_data.plot(x="timestamp", y="force")
    plt.show()
    exit()

    x0 = [min_c,k1,D,C,c0]
    start_time = time.time()
    result = differential_evolution(objective,x0 = x0, args = (X0, time_vector, active_force,m,km,kt,k2,A,B,max_c, sim_dt, reference_force), bounds = ((0.001, 10), (0.001, 10),(0.001, 10),(0.001, 5),(-5, 5)),disp= True)
    #result = minimize(objective, x0, args = (X0, time_vector, active_force,m,km,kt,k2,A,B,max_c, sim_dt, reference_force), method='L-BFGS-B',bounds = ((0.001, None), (0, None),(0, None),(0, None),(None, None)),options={'iprint':-1})#, jac=derivative)
    stop_time = time.time()
    print(result)
    print('Time: ', stop_time-start_time)
    fitted_muscle_data = rheopectic_muscle_response(X0,time_vector, active_force,m,km,kt,result.x[1],k2, A,B,result.x[3],result.x[2], max_c, result.x[0],result.x[4], sim_dt)
    plt.plot(time_vector, fitted_muscle_data['estimated force'])
    plt.legend(['Active force', 'Distribution of stimuli', 'Reference muscle force', 'Estimated muscle force'],loc ='upper right')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [mN]')
    #reference_force.plot(x='timestamp', y="reference force",ax=ax)
    plt.show()
    

    fitted_muscle_data = rheopectic_muscle_response(X0,time_vector, active_force,m,km,kt,4.99999969,k2, A,0.99999994,1.99999988,D, max_c, 0.03,c0, sim_dt)
    ax = fitted_muscle_data.plot(x='timestamp', y="estimated force")
    #plt.plot(time_vector, fitted_muscle_data['estimated force'])
    plt.plot(time_vector, active_force)
    reference_force.plot(x='timestamp', y="reference force",ax=ax)
    plt.legend(['Active force', 'Distribution of stimuli', 'Reference muscle force', 'Estimated muscle force'],loc ='upper right')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [mN]')
    plt.show()

    # 0.79420633,  0.49530833,  1.4736452 ,  0.54697721, -0.18024405
    # 0.79420638,  0.49530833,  1.47364526,  0.54697723, -0.18024402



muscle_state()
