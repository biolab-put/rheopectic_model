from time import time
from scipy.integrate import ode
import numpy as np

class hill_muscle_model:
    def __init__(self,km,kt,m,c,delta,sim_dt) -> None:
        self.km = km
        self.kt = kt
        self.m = m
        self.c = c
        self.delta = delta
        self.sim_dt = sim_dt

    def __solve_muscle_dynamics(self,t,X,active_force):
        lm, dlm_dt = X
        d2lm_dt = 1/self.m*(-self.c*dlm_dt-self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)])
        return [dlm_dt,d2lm_dt]

    def muscle_response(self,X0,time_vector,active_force):
        solver = ode(self.__solve_muscle_dynamics)
        solver.set_integrator('dopri5', atol = 1e-2,rtol=1e-2)
        solver.set_initial_value(X0)
        solver.set_f_params(active_force)
        solution = np.zeros((len(X0),len(time_vector)))
        solution[:,0] = X0
        for n in range(1,len(time_vector)):
            solver.integrate(time_vector[n])
            assert solver.successful()
            solution[:,n] = solver.y
        estimated_force = self.kt * np.sign(self.delta - solution[0,:]) * np.abs(self.delta - solution[0,:])
        estimated_force = estimated_force - estimated_force[0]
        return estimated_force

    def set_parameters(self,km,kt,m,c):
        self.km = km
        self.kt = kt
        self.m = m
        self.c = c

    def set_parameters(self,x):
        self.km = x[0]
        self.kt = x[1]
        self.m = x[2]
        self.c = x[3]

    def get_parameters(self):
        x = [self.km,self.kt,self.m,self.c]
        return x

    @staticmethod
    def get_damping_ratio(x):
        km = x[0]
        kt = x[1]
        m = x[2]
        c = x[3]
        natural_frequency = np.sqrt((km+kt)/m)
        damping_ratio = c / (2 * m * natural_frequency)
        return damping_ratio

    def get_initial_length(self):
        return self.kt * self.delta / (self.km + self.kt)



