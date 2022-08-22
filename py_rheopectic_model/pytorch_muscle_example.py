from torchdiffeq import odeint
import torch
from matplotlib import pyplot as plt 
import numpy as np

def first_order_example():
    def func(t,z):
            return t

    z0 = torch.Tensor([0])
    t = torch.linspace(0,2,100)
    out = odeint(func,z0,t)
    plt.plot(t,out)
    plt.legend(['dx/dt'])
    plt.show()

def second_order_example():
    class Oscillation:
        def __init__(self,km) -> None:
            self.mat = torch.Tensor([[0,1],[-km,0]])
        def solve(self,t,x0,dx0):
            y0 = torch.cat([x0,dx0])
            out = odeint(self.func,y0,t)
            return out
        def func(self,t,y):
            out = y @ self.mat #@is a matrix product
            return out
    x0 = torch.Tensor([1])
    dx0 = torch.Tensor([0])
    t = torch.linspace(0,4*np.pi,1000)
    solver = Oscillation(1)
    out = solver.solve(t,x0,dx0)
    plt.plot(t,out)
    plt.legend(['x: pos', 'dx/dt: vel'])
    plt.show()

def first_order_example_cuda():
    def func(t,z):
            return t
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Working on: ', device)
    z0 = torch.Tensor([0]).to(device)
    t = torch.linspace(0,2,100).to(device)
    out = odeint(func,z0,t).to(device)
    plt.plot(t.cpu(),out.cpu())
    plt.legend(['dx/dt'])
    plt.show()

def second_order_example_cuda():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Working on: ', device)
    def ode_sys(t,X):
        x = torch.Tensor([X[0]]).to(device)
        dx_dt = torch.Tensor([X[1]]).to(device)
        d2x_dt2 = torch.Tensor([-dx_dt-2*x]).to(device)
        return torch.cat([dx_dt,d2x_dt2]).to(device)
    
    t = np.linspace(0,12,100)
    x_init = torch.Tensor([1.]).to(device)
    dxdt_init = torch.Tensor([0.]).to(device)
    X0 = torch.cat([x_init,dxdt_init]).to(device)
    solution = odeint(ode_sys,X0,torch.Tensor(t).to(device)).to(device)
    plt.plot(t,solution.cpu())
    plt.legend(['x: pos', 'dx/dt: vel'])
    plt.show()

def muscle_example():

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

    class hill_muscle_model:

        def __init__(self,km,kt,m,c,delta,sim_dt) -> None:
            self.km = km
            self.kt = kt
            self.m = m
            self.c = c
            self.delta = delta
            self.sim_dt = sim_dt
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print('Working on: ', self.device)

        def muscle_response(self,X0,time_vector,active_force):
            self.active_force = active_force
            solution = odeint(self.__solve_muscle_dynamics,torch.Tensor(X0).to(self.device),torch.Tensor(time_vector).to(self.device),method='euler').to(self.device)
            estimated_force = self.kt * (self.delta - solution.cpu().numpy()[:,0])
            estimated_force = estimated_force - estimated_force[0]
            return estimated_force


        def __solve_muscle_dynamics(self,t,X):
            lm = torch.Tensor([X[0]]).to(self.device)
            dlm_dt = torch.Tensor([X[1]]).to(self.device)
            d2lm_dt2 = torch.Tensor([1/self.m*(-self.c*dlm_dt-self.km*lm+self.kt*(self.delta-lm)-self.active_force[int(t/self.sim_dt)])]).to(self.device)
            return torch.cat([dlm_dt,d2lm_dt2]).to(self.device)

    sim_dt = 0.0001
    sim_time = 0.8
    time_vector = np.arange(0,sim_time-sim_dt,sim_dt)
    twitch_duration = 0.005
    twitch_delay = 0.005
    twitch_frequency = 40
    twitch_amplitude = 1
    km = 0.1
    kt = 1
    m = 0.002
    c = 1 * 2 * m * np.sqrt((kt+km)/m)
    delta = 0
    X0 = [0,0]

    active_force = muscle_active_force.parabolic_twitch(time_vector,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
    active_force[int(0.525/sim_dt)::]=0
    #print(np.shape(time_vector))
    #print(np.shape(active_force))
    #exit()
    muscle_model = hill_muscle_model(km,kt,m,c,delta,sim_dt)
    muscle_force = muscle_model.muscle_response(X0,time_vector,active_force)
    plt.plot(time_vector,muscle_force)
    plt.show()

if __name__=="__main__":
    #first_order_example()
    #second_order_example()
    #first_order_example_cuda()
    #second_order_example_cuda()
    muscle_example()
