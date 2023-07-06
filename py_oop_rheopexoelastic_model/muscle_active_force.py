import numpy as np
from matplotlib import pyplot as plt 

class muscle_active_force:
        @staticmethod
        def _generate_wave(t,single_pulse,twitch_delay, twitch_frequency, sim_dt):
            delay = np.zeros(int(twitch_delay/sim_dt))
            single_twitch = np.concatenate([delay,single_pulse])
            if(1/twitch_frequency > t[-1]): # if there is only a single pulse
                single_twitch = np.pad(single_twitch, (0, t.shape[0] - single_twitch.shape[0]), 'constant', constant_values=0)
                return single_twitch
            else:
                T = 1/twitch_frequency
                twitches_mask = np.zeros(t.shape)
                twitches_mask[::int(T/sim_dt)] = 1
                twitch_train = np.zeros(t.shape) #initialize

                for i in range(len(twitches_mask)):
                    if i == 0:
                        twitch_train[i:len(single_twitch)] += single_twitch
                        #plt.plot(single_twitch)
                        #plt.plot(twitch_train)
                        pass
                    elif twitches_mask[i] == 1 and ((i + len(single_twitch)) < len(t)):
                        #print(i, len(t))
                        result = np.concatenate([np.zeros(i),single_twitch])
                        #print(result.shape)
                        twitch_train[0:len(result)] += result
                        #twitch_train[0:np.min([len(result),len(t)])] += result[0:np.min([len(result),len(t)])]
                        #twitch_train[i:i+len(single_twitch)] += single_twitch
                        #print(i,i+len(single_twitch))
                        #plt.plot(result)
                    elif twitches_mask[i] == 1 and ((i + len(single_twitch)) > len(t)):
                        result = np.concatenate([np.zeros(i),single_twitch])
                        #print((i + len(single_twitch)))
                        #plt.plot(result[0:len(t)])
                        twitch_train[(i + len(single_twitch))::] = result[(i + len(single_twitch))::]
                #plt.plot(twitch_train)
                #plt.plot(twitches_mask)
                #plt.show()
                return twitch_train
                plt.plot(twitch_train)
                plt.show()

            exit()
            print(1/twitch_frequency)
            plt.plot(single_twitch_t,single_pulse)
            plt.show()

            repeat = np.ceil((t[-1]-t[0])/(1/twitch_frequency))
            '''
            

            twitch_train_test = np.zeros(int(repeat) * int((1/twitch_frequency)/sim_dt) + len(single_pulse))

            for k in range(int(repeat)):
                 twitch_train_test[k * int((1/twitch_frequency)/sim_dt): k * int((1/twitch_frequency)/sim_dt) + len(single_pulse)] += single_pulse

            twitch_train_test = twitch_train_test[0:len(t)]
            twitch_train_test = np.roll(twitch_train_test,int((twitch_delay)/sim_dt))
            twitch_train_test[0:np.int32((twitch_delay)/sim_dt)] = 0
            '''
            #plt.plot(t,twitch_train_test)
            #plt.show()

            repeat = np.ceil((t[-1]-t[0])/(1/twitch_frequency))
            twitch_train = np.tile(single_pulse, int(repeat))
            twitch_train = twitch_train[0:len(t)]
            twitch_train = np.roll(twitch_train,int((twitch_delay)/sim_dt))
            twitch_train[0:np.int32((twitch_delay)/sim_dt)] = 0
            return twitch_train
        
        @classmethod
        def parabolic_twitches(cls,stim_dig,twitch_duration,twitch_delay,twitch_amplitude, sim_dt):
            single_twitch_t = np.arange(0,twitch_duration+sim_dt,sim_dt)
            single_pulse = twitch_amplitude * np.sin(np.pi * single_twitch_t / (twitch_duration))
            delay = np.zeros(int(twitch_delay/sim_dt))
            single_twitch = np.concatenate([delay,single_pulse])
            stimulation_indexes = np.where(stim_dig == 1)[0]
            twitch_train = 0 * np.arange(0,len(stim_dig)*sim_dt,sim_dt)
            for stimulation_index in stimulation_indexes:
                twitch_train[stimulation_index:stimulation_index+len(single_twitch)] += single_twitch
            return twitch_train
            #twitch_train[stimulation_indexes] += single_pulse
            plt.plot(twitch_train)
            #plt.plot(stim_dig)
            plt.show()
            exit()
            #twitch_train = cls._generate_wave(t,single_pulse,twitch_delay, twitch_frequency, sim_dt)

            #single_twitch_t = np.arange(0,twitch_duration,sim_dt)
            #full_twitch_t = np.arange(0,(1/twitch_frequency),sim_dt)
            #pulse = np.zeros(len(full_twitch_t))
            #single_pulse = twitch_amplitude * np.sin(np.pi * single_twitch_t / (twitch_duration))
            #pulse[0:len(single_twitch_t)] = single_pulse
            #twitch_train = cls._generate_wave(t,pulse,twitch_delay, twitch_frequency, sim_dt)

            return twitch_train
        
        @classmethod
        def parabolic_twitch(cls,t,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt):
            single_twitch_t = np.arange(0,twitch_duration+sim_dt,sim_dt)
            ##full_twitch_t = np.arange(0,(1/twitch_frequency),sim_dt)
            ##pulse = np.zeros(len(full_twitch_t))
            #pulse = np.zeros(len(single_twitch_t)) # dopisane
            single_pulse = twitch_amplitude * np.sin(np.pi * single_twitch_t / (twitch_duration))
            #pulse[0:len(single_twitch_t)] = single_pulse
            twitch_train = cls._generate_wave(t,single_pulse,twitch_delay, twitch_frequency, sim_dt)

            #single_twitch_t = np.arange(0,twitch_duration,sim_dt)
            #full_twitch_t = np.arange(0,(1/twitch_frequency),sim_dt)
            #pulse = np.zeros(len(full_twitch_t))
            #single_pulse = twitch_amplitude * np.sin(np.pi * single_twitch_t / (twitch_duration))
            #pulse[0:len(single_twitch_t)] = single_pulse
            #twitch_train = cls._generate_wave(t,pulse,twitch_delay, twitch_frequency, sim_dt)

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