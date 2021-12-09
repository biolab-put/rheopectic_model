import numpy as np

class muscle_active_force:
        @staticmethod
        def parabolic_twitch(t,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt):
            single_twitch_t = np.arange(0,twitch_duration,sim_dt)
            pulse = twitch_amplitude * np.sin(np.pi * single_twitch_t / twitch_duration)
            zeros_duration = (1/twitch_frequency) - twitch_duration
            assert zeros_duration > 0, "Twitch duration should be smaller than the inverse of it's frequency (period)."
            zeros = np.zeros(np.int32(zeros_duration/sim_dt))
            pulse = np.concatenate((pulse,zeros))
            #pulse = np.pad(pulse, (0, np.int32(((1/twitch_frequency) - twitch_duration)/sim_dt)), 'constant')
            repeat = np.ceil((t[-1]+sim_dt)/(1/twitch_frequency))
            twitch_train = np.tile(pulse, int(repeat))
            twitch_train = twitch_train[0:len(t)]
            twitch_train = np.roll(twitch_train,int((twitch_delay)/sim_dt))
            twitch_train[0:np.int32((twitch_delay+sim_dt)/sim_dt)] = 0
            return twitch_train
