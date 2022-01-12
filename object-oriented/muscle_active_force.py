import numpy as np
from matplotlib import pyplot as plt 

def inverse_rounding(number):
    return int(np.round(np.trunc(number)+ (1-np.mod(number, 1.0))))

class muscle_active_force:
        @staticmethod
        def parabolic_twitch(t,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt):
            #print('Twitch duration', 0.7 * twitch_duration)
            single_twitch_t = np.arange(0,twitch_duration,sim_dt)
            #duration = np.around((1/twitch_frequency),3)
            pulse = twitch_amplitude * np.sin(np.pi * single_twitch_t / (twitch_duration))
            #pulse = pulse * (1 / (1+np.exp(-(50/twitch_duration)*(single_twitch_t-0.4*twitch_duration)))) * (1 / (1+np.exp((50/twitch_duration)*(single_twitch_t-0.95*twitch_duration))))
            #pulse = np.roll(pulse, -int(0.3 * twitch_duration/sim_dt))
            #pulse[(int(0.95*twitch_duration/sim_dt))::] = 0
            #print(duration, twitch_duration)
            zeros_duration = (1/twitch_frequency) - (twitch_duration)
            #print(twitch_frequency,int(np.ceil(zeros_duration/sim_dt)))
            assert zeros_duration > 0, "Twitch duration should be smaller than the inverse of it's frequency (period)."
            #print(int(np.ceil(zeros_duration/sim_dt)))
            #print(zeros_duration/sim_dt,np.int32(zeros_duration/sim_dt))
            zeros = np.zeros(inverse_rounding(zeros_duration/sim_dt))
            #print(len(zeros))
            pulse = np.concatenate((pulse,zeros))
            #pulse = np.pad(pulse, (0, np.int32(((1/twitch_frequency) - twitch_duration)/sim_dt)), 'constant')
            repeat = np.ceil((t[-1]-t[0])/(1/twitch_frequency))
            twitch_train = np.tile(pulse, int(repeat))
            twitch_train = twitch_train[0:len(t)]
            twitch_train = np.roll(twitch_train,int((twitch_delay)/sim_dt))
            twitch_train[0:np.int32((twitch_delay)/sim_dt)] = 0
            return twitch_train

        def sigmoid_twitch(t,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt):
            single_twitch_t = np.arange(0,twitch_duration+twitch_delay,sim_dt)
            #pulse =  twitch_amplitude*((1/ (1+np.exp(-10000*(single_twitch_t)))) + (0.5/ (1+np.exp(1000*(single_twitch_t-twitch_duration)))))-twitch_amplitude
            pulse = twitch_amplitude * (1/(1+np.exp(-10000*(single_twitch_t-twitch_delay-twitch_delay/2.5)))) * (1/(1+(np.exp(500*(single_twitch_t-twitch_duration-twitch_delay)))))
            #pulse = pulse[pulse>=0.75*twitch_amplitude]
            plt.plot(pulse)
            plt.show()
            exit()

        def exp_twitch(t,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt):
            single_twitch_t = np.arange(0,twitch_duration,sim_dt)
            pulse = (1 - np.exp(-single_twitch_t/0.00005))*(1/(1+np.exp(700*(single_twitch_t-twitch_duration/10))))
            #pulse = (1 - np.exp(-single_twitch_t/0.00005))*(1/(1+np.exp(1900*(single_twitch_t-twitch_duration/2))))
            zeros_duration = (1/twitch_frequency) - twitch_duration
            assert zeros_duration > 0, "Twitch duration should be smaller than the inverse of it's frequency (period)."
            zeros = np.zeros(np.int32(zeros_duration/sim_dt))
            pulse = np.concatenate((pulse,zeros))
            repeat = np.ceil((t[-1]+sim_dt)/(1/twitch_frequency))
            twitch_train = np.tile(pulse, int(repeat))
            twitch_train = twitch_train[0:len(t)]
            twitch_train = np.roll(twitch_train,int((twitch_delay)/sim_dt))
            twitch_train[0:np.int32((twitch_delay+sim_dt)/sim_dt)] = 0
            return twitch_train
