import numpy as np
from hill_muscle_model import rheopectic_hill_muscle_model, hill_muscle_model
from muscle_active_force import muscle_active_force
from muscle_simulation import prepare_reference_data_in
from matplotlib import pyplot as plt 
from hill_muscle_model import rheopectic_modified_hill_muscle_model

def fun(a,b):
    print(a)
    print(b)
    return a+b

def create_simple_param_grid(points_per_parameter):
    AA = np.linspace(0,1.0,points_per_parameter)
    #AA = np.log(3*AA+1)
    #AA = AA ** (2)
    BB = np.linspace(1,20,points_per_parameter)
    #BB = np.x(3*BB+1)
    #BB = BB ** 2
    CC = np.linspace(1,20,points_per_parameter)
    #CC = np.log(3*CC+1)
    #CC = CC ** 2
    DD = np.linspace(0,20,points_per_parameter)
    #DD = np.log(3*DD+1)
    #DD = DD ** 2
    kk1 = np.linspace(0.1,100,points_per_parameter)
    #kk1 = np.log(3*kk1+1)
    #kk1 = kk1 ** 2
    kk2 = np.linspace(0.1,100,points_per_parameter)
    #kk2 = np.log(3*kk2+1)
    #kk2 = kk2 ** 2
    #min_cc = np.linspace(0.1,0.5,points_per_parameter)
    #min_cc = np.log(3*min_cc+1)
    #min_cc = min_cc ** 2
    #max_cc = np.linspace(0.99,15,points_per_parameter)
    #max_cc = np.log(3*max_cc+1)
    #max_cc = max_cc ** 2
    #cc0 = np.linspace(-0.65,0.65,points_per_parameter)
    c_rh_start = 2.10558506e+01
    c_rh_min = 1.0490643e+00
    c1 = 4.30490643e+01
    cs = 1.35102053e+01 

    km = 8.21688727e+00 
    kt = 5.45036325e+01
    m = 2.86717809e-02
    kmm = np.linspace(km/10 ,km *10,points_per_parameter)
    ktt = np.linspace(kt/1.2 ,kt *5,points_per_parameter)
    mm = np.linspace(m/50 ,m * 1.5,points_per_parameter)

    
    c_rhh = np.linspace(c_rh_start/5 ,c_rh_start *5,points_per_parameter)
    c_rh_minn = np.linspace(c_rh_min/5 ,c_rh_min*5,points_per_parameter)
    c11 = np.linspace(c1/10 ,c1*10,points_per_parameter)
    css = np.linspace(cs / 10, cs * 10,points_per_parameter)
    #cc0 = np.log(3*cc0+1)
    lambda00 = np.linspace(0.05,0.98,points_per_parameter)
    
    kss = np.linspace(0.1,100,points_per_parameter)
    ls00 = np.linspace(-0.01,0.01,points_per_parameter)
    F00 = np.linspace(-0.05,0.05,points_per_parameter)
    #param_grid = {'A': AA, 'B': BB, 'C': CC, 'D' : DD, 'k1' : kk1, 'k2' : kk2, 'km' : kmm,'kt' : ktt, 'm' : mm, 'lambda0' : lambda00, 'F0': F00}
    #param_grid = {'km' : kmm,'kt' : ktt, 'm' : mm}
    param_grid = {'c1' : c11,'c_rh' : c_rhh, 'c_rh_min' : c_rh_minn,'cs' : css}
    return param_grid

def create_param_grid(points_per_parameter):
    AA = np.linspace(0,1,points_per_parameter)
    BB = np.linspace(0,1,points_per_parameter)
    CC = np.linspace(1,5,points_per_parameter)
    DD = np.linspace(0,5,points_per_parameter)
    kk1 = np.linspace(0.01,5,points_per_parameter)
    kk2 = np.linspace(0.01,5,points_per_parameter)
    min_cc = np.linspace(0.001,0.5,points_per_parameter)
    max_cc = np.linspace(0.99,15,points_per_parameter)
    cc0 = np.linspace(-0.03,0.015,points_per_parameter)
    param_grid = {'A': AA, 'B': BB, 'C': CC, 'D' : DD, 'k1' : kk1, 'k2' : kk2, 'min_c' : min_cc, 'max_c' : max_cc, 'c0' : cc0}
    print(np.shape(list(ParameterGrid(param_grid))))


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

    muscle_model = rheopectic_modified_hill_muscle_model(km,kt,m,cs,ks,ls0,c_rh,c_rh_min,c1,c2,k1,k2,A,B,C,D,lambda0,F0,delta,sim_dt)
    return muscle_model

def rheopectic_modified_muscle_grid_search(parameters):

    '''
    sim_dt = 0.0001
    twitch_duration = 0.01
    twitch_delay = 0.0025
    twitch_frequency = 40
    low_frequency = 35
    high_frequency = 75
    twitch_amplitude = 1
    #km = 3.10865655e+00
    #kt = 4.06278815e+01
    #m = 2.08453333e-02

    km = 7.05745487e-01 * 1
    kt = 7.06960318e+01 * 1 * 2
    m = 2.28151620e-02 
    c = 20.65682892e+00 * 10
    c_rh_max = c / 4
    c_rh_min = 5.37
    c1 = c  / 5
    cs = 1.35102053e+01 * 2
    ks = 2.45926152e+02 * 1
    ls0 = -0.0025
    k1 = 10.46033652e+00 * 30
    k2 = 9.31295195e+01 * 31
    A = 3
    B = 1
    C = 1
    D = 1
    lambda0 = 6.11381772e-01
    delta = 0.01
    F0 = 0 
    c0 = 0.01
    #lambda0 = 8.50139814e-01
    '''
    sim_dt = 0.0001
    twitch_duration = 0.01
    twitch_delay = 0.0025
    twitch_frequency = 40
    low_frequency = 35
    high_frequency = 75
    twitch_amplitude = 1

    muscle_model = create_rheopectic_modified_model(sim_dt)
    #muscle_model = rheopectic_modified_hill_muscle_model(km,kt,m,cs,ks,ls0,c_rh,c_rh_min,c1,c2,k1,k2,A,B,C,D,lambda0,F0,delta,sim_dt)

    #startowac z wiekszej lambdy,nasycenie obserwowac na wykresie lambdy, dodac wykres lambdy
    #muscle_model = hill_muscle_model(km,kt,m,2.27624249e+00,0,0,delta,sim_dt)
    #muscle_model = rheopectic_modified_hill_muscle_model(km,kt,m,cs,ks,ls0,c0,k1,k2,A,B,C,D,lambda0,delta,sim_dt)
    #muscle_model = rheopectic_modified_hill_muscle_model(km,kt,m,cs,ks,ls0,c_rh_max,c_rh_min,c0,c1,k1,k2,A,B,C,D,lambda0,F0,delta,sim_dt)
    filenames = ['947kLHLf/947kLHLf_trial_07','947kLHLf/947kLHLf_trial_17'] #17
    # 15 wykres grid search TODO <--------------------
    reference_force_0 = prepare_reference_data_in(filenames[0],sim_dt,steady_time=1.5,zeros_time=0.3)
    reference_force_1 = prepare_reference_data_in(filenames[1],sim_dt,steady_time=1.5,zeros_time=0.3)
    reference_forces = [reference_force_0['force'],reference_force_1['force']]
    time_vector_0 = np.arange(0,len(reference_forces[0])*sim_dt,sim_dt)
    time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    #reference_force_0['stimDig']
    #plt.plot(time_vector_0,reference_force_0['force'])
    #plt.plot(time_vector_1,reference_force_1['stimDig'])
    #plt.show()
    #exit()
    #there was some delay during switching frequency so I had to minus sim_dt/1.5
    low_freq_duration = int((19 * ((1/low_frequency - sim_dt/1.5)))/sim_dt)
    
    high_freq_duration = int((23 * ((1/high_frequency + sim_dt/1.5)))/sim_dt)
    #t =np.array([0,1,2,3,4])
    #a = np.array([1,0,0,1,0])
    #print(t[0:3])
    #print(t[3::])
    #exit()
    #print(low_freq_duration*sim_dt)
    #print((low_freq_duration+high_freq_duration)*sim_dt)
    #print((2*low_freq_duration+high_freq_duration)*sim_dt)
    input_1_low_freq = muscle_active_force.parabolic_twitch(time_vector_1[0:low_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, low_frequency, sim_dt)
    input_1_high_freq = muscle_active_force.parabolic_twitch(time_vector_1[low_freq_duration:low_freq_duration+high_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, high_frequency, sim_dt)
    input_1_low_freq_end = muscle_active_force.parabolic_twitch(time_vector_1[low_freq_duration+high_freq_duration:2*low_freq_duration+high_freq_duration],twitch_duration,twitch_delay,twitch_amplitude, low_frequency, sim_dt)

    input_1a = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    input_1a[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal

    #plt.plot(time_vector_1[0:low_freq_duration],input_1_low_freq)
    #plt.plot(time_vector_1[low_freq_duration:low_freq_duration+high_freq_duration],input_1_high_freq)
    #plt.step(time_vector_1[0:low_freq_duration],reference_force_1['stimDig'][0:low_freq_duration])
    #plt.step(time_vector_1[low_freq_duration:low_freq_duration+high_freq_duration],reference_force_1['stimDig'][low_freq_duration:low_freq_duration+high_freq_duration])
    
    #plt.plot(time_vector_1[(low_freq_duration+high_freq_duration)::],input_1_low_freq_end)
    #plt.show()
    #exit()
    zeros = np.zeros([len(time_vector_1)-(2*low_freq_duration+high_freq_duration)])
    input_1 = np.concatenate((input_1_low_freq, input_1_high_freq,input_1_low_freq_end,zeros), axis=0)
    #plt.plot(time_vector_1[0:low_freq_duration],reference_force_1['stimDig'][0:low_freq_duration])
    #plt.plot(time_vector_1[0:low_freq_duration],input_1)
    #plt.plot(input_1_high_freq)
    #print(len(input_1))
    # INPUT
    #plt.plot(time_vector_1,input_1)
    #plt.step(time_vector_1,reference_force_1['stimDig'])



    reference_stim_dig = np.copy(reference_force_1['stimDig'])
    reference_stim_dig[reference_stim_dig == 0] = np.nan

    # OUTPUT
    X0 = muscle_model.get_X0()
    muscle_data,[lm,dlm_dt,Lambda,ls] = muscle_model.muscle_response(X0,time_vector_1,input_1)
    plt.plot(time_vector_1,muscle_data)
    plt.plot(time_vector_1,reference_forces[1])
    plt.show()
    #exit()    

    #plt.figure(1)
    #plt.plot(time_vector_1,muscle_data,'tab:blue')
    #plt.plot(time_vector_1,reference_force_1['force'], 'tab:orange')
    #reference_stim_dig = np.copy(reference_force_1['stimDig'])
    #reference_stim_dig[reference_stim_dig == 0] = np.nan
    #plt.scatter(time_vector_0,reference_stim_dig * 0,2,'black',alpha = 1)
    #plt.grid(color='grey', linestyle='-', linewidth=0.3)
    #plt.xlabel('Time [s]')
    #plt.ylabel('Force [N]')
    #plt.show()
    #exit()
    #fig, axs = plt.subplots(nrows=2, ncols=1)
    #axs[0].plot(time_vector_1,Lambda)
    #axs[1].plot(time_vector_1,dlm_dt)
    #fig.supxlabel('Time [s]', fontsize = 'medium')
    #axs[0].set_ylabel('Lambda [-]')
    #axs[1].set_ylabel('dlm_dt [m/s]')
    #y_bottom,y_top= axs[0].get_ylim()
    #axs[0].scatter(time_vector_1,reference_stim_dig * y_bottom,2,'black',alpha = 1)
    #y_bottom,y_top= axs[1].get_ylim()
    #axs[1].scatter(time_vector_1,reference_stim_dig * y_bottom,2,'black',alpha = 1)
    #axs[0].grid(color='grey', linestyle='-', linewidth=0.3)
    #axs[1].grid(color='grey', linestyle='-', linewidth=0.3)


    #K = 1 - np.sqrt(max_c/min_c)
    #c = max_c/((1 - K * Lambda)**2) + c0
    #print((1-np.sqrt(max_c/c[0]))/(1-np.sqrt(max_c/min_c)))
    #plt.figure(3)
    #plt.scatter(lm[int(0.55/sim_dt):int(0.7/sim_dt)],(c*dlm_dt)[int(0.55/sim_dt):int(0.7/sim_dt)],2,'tab:blue','o')
    #plt.scatter(lm[int(0.85/sim_dt):int(1.0/sim_dt)],(c*dlm_dt)[int(0.85/sim_dt):int(1.0/sim_dt)],2,'tab:orange','o')
    #plt.xlabel('lm [m]')
    #plt.ylabel('Force [N]')
    #plt.figure(4)
    #plt.scatter(lm[int(0.55/sim_dt):int(0.7/sim_dt)],(muscle_data)[int(0.55/sim_dt):int(0.7/sim_dt)],2,'tab:blue','o')
    #plt.scatter(time_vector_1,(c*dlm_dt),2,'tab:orange','o')

    #plt.figure(5)
    #plt.scatter(lm[int(0.55/sim_dt):int(0.7/sim_dt)],(muscle_data)[int(0.55/sim_dt):int(0.7/sim_dt)],2,'tab:blue','o')
    #plt.scatter(time_vector_1,(muscle_data),2,'tab:orange','o')

    
    #plt.show()
    #exit()
    
    #int(((19 * ((1/twitch_frequency))) + (24 * ((1/75))))/sim_dt)
    #input_1 = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    #input_1[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal

    #X0 = muscle_model.get_X0()
    #muscle_data = muscle_model.muscle_response(X0,time_vector_1,input_1)
    #plt.plot(time_vector_1,reference_forces[1])
    #plt.show()
    #exit()


    fig, axs = plt.subplots(sharex=False, nrows=int(np.ceil(len(parameters)/2)), ncols=2,constrained_layout=True,figsize=(10,7))
    fig_lambda, axs_lambda = plt.subplots(sharex=False, nrows=int(np.ceil(len(parameters)/2)), ncols=2,constrained_layout=True,figsize=(10,7))

    print('Grid search status: ')
    for param_index, (parameter,values) in enumerate(parameters.items()):
        print(f'Parameter {param_index+1}/{len(parameters)}: {parameter}')
        for value_index, value in enumerate(values):
            print(f'Value {value_index+1}/{len(values)}')
            initial_parameter_value = muscle_model.set_parameter(parameter,value)
            X0 = muscle_model.get_X0()
            muscle_data,[lm,dlm_dt,Lambda,ls] = muscle_model.muscle_response(X0,time_vector_1,input_1)
            #plt.plot(time_vector_1,muscle_data)
            #plt.title(f'{parameter} = {value:,.2f}')
            #plt.show()
            axs[int(np.floor(param_index/2)), param_index%2].plot(time_vector_1,muscle_data,label=f'{parameter} = {value:,.2f}')
            muscle_model.set_parameter(parameter,initial_parameter_value)
            axs[int(np.floor(param_index/2)), param_index%2].set_title(parameter, fontsize = 10)
            axs[int(np.floor(param_index/2)), param_index%2].legend(fontsize=6, loc = 'right',framealpha=0.7)

            axs_lambda[int(np.floor(param_index/2)), param_index%2].plot(time_vector_1,Lambda,label=f'{parameter} = {value:,.2f}')
            axs_lambda[int(np.floor(param_index/2)), param_index%2].set_title(parameter, fontsize = 10)
            axs_lambda[int(np.floor(param_index/2)), param_index%2].legend(fontsize=6, loc = 'right',framealpha=0.7)
            

    #axs[-1,0].set_xlabel('Time [s]')
    #axs[-1,1].set_xlabel('Time [s]')
        #break
    fig.supylabel('Force [N]', fontsize = 'medium')
    fig.supxlabel('Time [s]', fontsize = 'medium')
    fig_lambda.supylabel('Lambda [-]', fontsize = 'medium')
    fig_lambda.supxlabel('Time [s]', fontsize = 'medium')
    plt.show()
    exit()
    # TODO
    # Wykres Lambdy
    # Lambda, nasycenie po całkowaniu porównać z nasyceniem w obliczeniach

parameters = create_simple_param_grid(5)
rheopectic_modified_muscle_grid_search(parameters)
#print(fun(x,y))
