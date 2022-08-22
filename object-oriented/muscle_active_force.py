import numpy as np

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
            single_pulse[np.arange(m/sim_dt,a/sim_dt,1,dtype = int)] = twitch_amplitude * (single_twitch_t[np.arange(m/sim_dt,a/sim_dt,1,dtype = int)] - m) / (a - m)
            single_pulse[np.arange(b/sim_dt,r/sim_dt,1,dtype = int)] = twitch_amplitude * (r - single_twitch_t[np.arange(b/sim_dt,r/sim_dt,1,dtype = int)]) / (r - b)
            single_pulse[np.arange(a/sim_dt,b/sim_dt,1,dtype = int)] = twitch_amplitude
            pulse[0:len(single_twitch_t)] = single_pulse
            twitch_train = cls._generate_wave(t,pulse,twitch_delay, twitch_frequency, sim_dt)
            return twitch_train
