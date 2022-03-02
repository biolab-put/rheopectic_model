from re import M
import numpy as np
from matplotlib import pyplot as plt 

def inverse_rounding(number):
    return int(np.round(np.trunc(number)+ (1-np.mod(number, 1.0))))

class muscle_active_force:
        @staticmethod
        def _generate_wave(t,single_pulse,twitch_delay, twitch_frequency, sim_dt):
            repeat = np.ceil((t[-1]-t[0])/(1/twitch_frequency))
            twitch_train = np.tile(single_pulse, int(repeat))
            twitch_train = twitch_train[0:len(t)]
            twitch_train = np.roll(twitch_train,int((twitch_delay)/sim_dt))
            twitch_train[0:np.int32((twitch_delay)/sim_dt)] = 0
            return twitch_train

        @classmethod
        def parabolic_twitch(cls,t,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt):
            single_twitch_t = np.arange(0,twitch_duration,sim_dt)
            full_twitch_t = np.arange(0,(1/twitch_frequency),sim_dt)
            pulse = np.zeros(len(full_twitch_t))
            single_pulse = twitch_amplitude * np.sin(np.pi * single_twitch_t / (twitch_duration))
            pulse[0:len(single_twitch_t)] = single_pulse
            twitch_train = cls._generate_wave(t,pulse,twitch_delay, twitch_frequency, sim_dt)
            return twitch_train

        @classmethod
        def square_twitch(cls,t,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt):
            single_twitch_t = np.arange(0,twitch_duration,sim_dt)
            full_twitch_t = np.arange(0,(1/twitch_frequency),sim_dt)
            pulse = np.zeros(len(full_twitch_t))
            single_pulse = twitch_amplitude * np.ones(np.shape(single_twitch_t))
            pulse[0:len(single_twitch_t)] = single_pulse
            #pulse[0] = 0
            twitch_train = cls._generate_wave(t,pulse,twitch_delay, twitch_frequency, sim_dt)
            return twitch_train

        @classmethod
        def trapezoidal_twitch(cls,t,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt):
            m = 0
            r = twitch_duration
            a = m + twitch_duration/10
            b = r - twitch_duration/10
            single_twitch_t = np.arange(0,twitch_duration,sim_dt)
            full_twitch_t = np.arange(0,(1/twitch_frequency),sim_dt)
            pulse = np.zeros(len(full_twitch_t))
            single_pulse = np.zeros(len(single_twitch_t))
            single_pulse[np.arange(m/sim_dt,a/sim_dt,1,dtype = int)] = (single_twitch_t[np.arange(m/sim_dt,a/sim_dt,1,dtype = int)] - m) / (a - m)
            single_pulse[np.arange(b/sim_dt,r/sim_dt,1,dtype = int)] = (r - single_twitch_t[np.arange(b/sim_dt,r/sim_dt,1,dtype = int)]) / (r - b)
            single_pulse[np.arange(a/sim_dt,b/sim_dt,1,dtype = int)] = twitch_amplitude
            pulse[0:len(single_twitch_t)] = single_pulse
            twitch_train = cls._generate_wave(t,pulse,twitch_delay, twitch_frequency, sim_dt)
            return twitch_train
        
        @classmethod
        def exp_twitch(cls,t,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt):
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
