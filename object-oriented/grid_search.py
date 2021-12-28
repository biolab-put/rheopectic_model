import numpy as np
from hill_muscle_model import rheopectic_hill_muscle_model
from muscle_active_force import muscle_active_force
from muscle_simulation import prepare_reference_data
from matplotlib import pyplot as plt 

def fun(a,b):
    print(a)
    print(b)
    return a+b

def create_simple_param_grid(points_per_parameter):
    AA = np.linspace(0,2.0,points_per_parameter)
    #AA = np.log(3*AA+1)
    AA = AA ** (2)
    BB = np.linspace(0,2,points_per_parameter)
    #BB = np.x(3*BB+1)
    BB = BB ** 2
    CC = np.linspace(1,5,points_per_parameter)
    #CC = np.log(3*CC+1)
    CC = CC ** 2
    DD = np.linspace(0,5,points_per_parameter)
    #DD = np.log(3*DD+1)
    DD = DD ** 2
    kk1 = np.linspace(0.01,50,points_per_parameter)
    #kk1 = np.log(3*kk1+1)
    kk1 = kk1 ** 2
    kk2 = np.linspace(0.01,50,points_per_parameter)
    #kk2 = np.log(3*kk2+1)
    kk2 = kk2 ** 2
    min_cc = np.linspace(1,5,points_per_parameter)
    #min_cc = np.log(3*min_cc+1)
    min_cc = min_cc ** 2
    max_cc = np.linspace(10,150,points_per_parameter)
    #max_cc = np.log(3*max_cc+1)
    max_cc = max_cc ** 2
    cc0 = np.linspace(-0.1,15,points_per_parameter)
    #cc0 = np.log(3*cc0+1)
    lambda00 = np.linspace(0,1,points_per_parameter)
    param_grid = {'A': AA, 'B': BB, 'C': CC, 'D' : DD, 'k1' : kk1, 'k2' : kk2, 'min_c' : min_cc, 'max_c' : max_cc, 'c0' : cc0, 'lambda0' : lambda00}
    return param_grid

def create_param_grid(points_per_parameter):
    AA = np.linspace(0,1,points_per_parameter)
    BB = np.linspace(0,1,points_per_parameter)
    CC = np.linspace(1,5,points_per_parameter)
    DD = np.linspace(0,5,points_per_parameter)
    kk1 = np.linspace(0.01,5,points_per_parameter)
    kk2 = np.linspace(0.01,5,points_per_parameter)
    min_cc = np.linspace(0.001,50,points_per_parameter)
    max_cc = np.linspace(0.1,150,points_per_parameter)
    cc0 = np.linspace(-0.03,0.015,points_per_parameter)
    param_grid = {'A': AA, 'B': BB, 'C': CC, 'D' : DD, 'k1' : kk1, 'k2' : kk2, 'min_c' : min_cc, 'max_c' : max_cc, 'c0' : cc0}
    print(np.shape(list(ParameterGrid(param_grid))))


def rheopectic_muscle_grid_search(parameters):
    sim_dt = 0.0001
    twitch_duration = 0.015
    twitch_delay = 0.0025
    twitch_frequency = 40
    twitch_amplitude = 1
    km = 3.10865655e+00
    kt = 4.06278815e+01
    m = 2.08453333e-02
    #c = 1 * 2 * m * np.sqrt((kt+km)/m)
    delta = 0.01
    #F_K = 1
    #F_delta = 1
    c0 = 0
    min_c = 1
    max_c = 1200
    k1 = 5
    k2 = 40
    A = 0
    B = 1.5
    C = 2
    D = 2
    lambda0 = 0.5
    muscle_model = rheopectic_hill_muscle_model(km,kt,m,c0,min_c,max_c,k1,k2,A,B,C,D,lambda0,delta,sim_dt)
    filenames = ['1006j_trial_04','1006j_trial_05']
    reference_forces = prepare_reference_data(filenames,sim_dt)
    time_vector_1 = np.arange(0,len(reference_forces[1])*sim_dt,sim_dt)
    input_1 = muscle_active_force.parabolic_twitch(time_vector_1,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    input_1[int(0.525/sim_dt)::]=0 # to be the same like the orginal signal

    #X0 = muscle_model.get_X0()
    #muscle_data = muscle_model.muscle_response(X0,time_vector_1,input_1)
    #plt.plot(time_vector_1,muscle_data)
    #plt.show()
    #exit()

    fig, axs = plt.subplots(sharex=False, nrows=int(len(parameters)/2), ncols=2,constrained_layout=True,figsize=(10,7))

    print('Grid search status: ')
    for param_index, (parameter,values) in enumerate(parameters.items()):
        print(f'Parameter {param_index+1}/{len(parameters)}: {parameter}')
        for value in values:
            initial_parameter_value = muscle_model.set_parameter(parameter,value)
            X0 = muscle_model.get_X0()
            muscle_data = muscle_model.muscle_response(X0,time_vector_1,input_1)
            axs[int(np.floor(param_index/2)), param_index%2].plot(time_vector_1,muscle_data,label=f'{parameter} = {value:,.2f}')
            muscle_model.set_parameter(parameter,initial_parameter_value)
            axs[int(np.floor(param_index/2)), param_index%2].set_title(parameter, fontsize = 10)
            axs[int(np.floor(param_index/2)), param_index%2].legend(fontsize=6, loc = 'right',framealpha=0.7)
    #axs[-1,0].set_xlabel('Time [s]')
    #axs[-1,1].set_xlabel('Time [s]')
        #break
    fig.supylabel('Force [N]', fontsize = 'medium')
    fig.supxlabel('Time [s]', fontsize = 'medium')
    plt.show()

parameters = create_simple_param_grid(5)
rheopectic_muscle_grid_search(parameters)
#print(fun(x,y))
