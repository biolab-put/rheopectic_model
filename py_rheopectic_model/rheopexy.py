from re import A
import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt 
from scipy import signal
from muscle_active_force import muscle_active_force
from scipy.integrate import odeint
from scipy.integrate import solve_ivp



class rheopexy:
    def __init__(self,c0,min_c,max_c,k1,k2,A,B,C,D,sim_dt) -> None:
        self.k1 = k1
        self.k2 = k2
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.c0 = c0
        self.min_c = min_c
        self.max_c = max_c
        self.sim_dt = sim_dt

    def _solve_lambda_dynamics(self,t,X,input):
        Lambda = X
        Lambda_temp = np.max([0.0, np.min([1.0, Lambda])])
        dlm_dt = input[int(t/self.sim_dt)]
        
        #dLambda_dt = -self9.k1*np.sign(dlm_dt)*((np.abs(dlm_dt))**(self.A))*(Lambda_temp**(self.B))+self.k2*np.sign(dlm_dt)*((np.abs(dlm_dt))**(self.C))*((1-Lambda_temp)**(self.D))
        dLambda_dt = -self.k1 * (dlm_dt**self.A) * (Lambda_temp**self.B) + self.k2 *(dlm_dt**self.C) * (1-Lambda_temp)**(self.D)
        if(Lambda >= 1.0 and dLambda_dt > 0.0):
            Lambda = 1.0
            dLambda_dt = 0.0
        elif(Lambda <= 0.0 and dLambda_dt < 0.0):
            Lambda = 0.0
            dLambda_dt = 0.0
        ## nasyca sie przez ograniczenie <0,1>
        return dLambda_dt

    def lambda_response(self,X0,time_vector,input):
        #solution = solve_ivp(fun = self._solve_lambda_dynamics,method = 'RK45',t_span = (time_vector[0],time_vector[-1]), y0 = X0,t_eval = time_vector, args = (input,),first_step=1e-2*0.0001,max_step =1e-2*0.0001, atol=1e-2*0.0001)
        #print(solution.message)
        solver = ode(self._solve_lambda_dynamics)
        solver.set_integrator('dopri5', atol = 1e-5,rtol=1e-5)
        solver.set_initial_value(X0)
        solver.set_f_params(input)
        solution = np.zeros((len(X0),len(time_vector)))
        solution[:,0] = X0
        for n in range(1,len(time_vector)):
            solver.integrate(time_vector[n])
            assert solver.successful()
            solution[:,n] = solver.y
        return solution


def test():
            # function that returns dy/dt
    def model(t,y):
        k = 0.3
        dydt = -k * y
        return dydt

    # initial condition
    y0 = 5

    # time points
    time_vector = np.linspace(0,20)

    solver = ode(model)
    solver.set_integrator('dopri5', atol = 1e-2, rtol=1e-2, max_step = 1e-5)
    X0 = [y0]
    solution = np.zeros((len(X0),len(time_vector)))
    solution[:,0] = X0
    for n in range(1,len(time_vector)):
        solver.integrate(time_vector[n])
        solution[:,n] = solver.y

    #solution = odeint(model,y0,time_vector,tfirst = True)
    #plt.plot(time_vector,solution)
    #solution = solve_ivp(model,(time_vector[0],time_vector[-1]),[y0],t_eval = time_vector)
    plt.plot(time_vector,solution[0,:])
    plt.show()
    exit()

if __name__=="__main__":
    c0 = 44
    min_c = 0.30
    max_c = 70
    k1 = 50
    k2 = 100
    A = 1
    B = 1
    C = 3
    D = 1
    dlm0 = 1
    dlm1 = 0.8
    temp = k2/k1 * dlm0**(C-A)
    lambda0 = temp/(1+temp)
    print(lambda0)
    sim_dt = 0.0001
    rheopectic_model = rheopexy(c0,min_c,max_c,k1,k2,A,B,C,D,sim_dt)
    X0 = [lambda0]
    time_duration = 0.1
    #time_vector = np.arange(0,time_duration,sim_dt)
    #ones = dlm0 * np.ones(int(0.1/sim_dt))
    #zeros = dlm0 * np.ones(int(0.1/sim_dt))
    
    #dlm_dt = np.concatenate((zeros,ones, dlm1 * zeros,ones))
    #last_zeros = int(0.1/sim_dt)
    #
    #dlm_dt = np.concatenate((dlm_dt,dlm0 * np.ones(last_zeros)))
 
    
    time_vector = np.arange(0,time_duration,sim_dt)
    #input_0 = muscle_active_force.parabolic_twitch(time_vector,0.05,0,1, 10, sim_dt)
    input_0 = muscle_active_force.trapezoidal_twitch(time_vector,0.01,0.002,1, 10, sim_dt)
    dlm_dt = input_0 + dlm0
    
    plt.plot(time_vector,dlm_dt)
    plt.xlabel('Time [s]')
    plt.ylabel(r'$\dot{l_{m}} \left[\frac{m}{s}\right]$')
    plt.grid()
    #plt.show()
    #exit()

    solution = rheopectic_model.lambda_response(X0,time_vector,dlm_dt)
    shear_stress = solution[0,:] * dlm_dt * 1
    plt.figure()
    plt.xlabel('Time [s]')
    plt.ylabel(r'$\lambda$ [-]')
    plt.plot(time_vector,solution[0,:])
    plt.grid()

    plt.figure()
    #plt.scatter(dlm_dt,shear_stress)

    # decrease
    plt.scatter(dlm_dt[np.append(np.diff(shear_stress) < 0,True)],shear_stress[np.append(np.diff(shear_stress) < 0,True)],color = 'blue',s = 20)
    # rise
    plt.scatter(dlm_dt[np.append(np.diff(shear_stress) > 0,True)],shear_stress[np.append(np.diff(shear_stress) > 0,True)], color = 'orange',s = 20)

    plt.xlabel(r'$\dot{l_{m}} \left[\frac{m}{s}\right]$')
    plt.ylabel(r'F [N]')
    plt.grid()
    plt.show()
