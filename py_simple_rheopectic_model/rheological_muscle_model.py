from scipy.integrate import ode
import numpy as np


class RheopecticMuscle():
 # TODO: OVERWRITE METHODS
    def __init__(self, km,kt,m,cs,ks,ls0,c_rh,c_rh_min,c1,k1,k2,lambda0,F0,s1,delta,sim_dt):
        self.k1 = k1
        self.k2 = k2
        self.m = m
        self.km = km
        self.kt = kt
        self.cs = cs
        self.ks = ks
        self.ls0 = ls0
        self.c1 = c1
        self.lambda0 = lambda0
        self.c_rh = c_rh
        self.c_rh_min = c_rh_min
        self.F0 = F0
        self.s1 = s1
        self.delta = delta
        self.sim_dt = sim_dt

    def _get_dls_dt(self, current_ls, current_active_force):
        dls_dt = 1/self.cs * (-(self.ks * current_ls) + current_active_force)
        return dls_dt

    def _solve_muscle_dynamics(self,t,X,active_force):
        lm, dlm_dt, Lambda,ls = X
        Lambda = np.clip(Lambda,0,1)
        #dlm_dt_temp = np.max([0,dlm_dt])

        dGamma_dt = np.max([0,dlm_dt]) / self.s1

        #print( - self.k1 * (Lambda), self.k2 * dGamma_dt * (1-Lambda) )
        dLambda_dt = - self.k1 * (Lambda) + self.k2 * dGamma_dt * (1-Lambda)
        #if(dLambda_dt < 1e-1):
        #    Lambda = ((self.k2 * dGamma_dt) / self.k1) / ( 1 + ((self.k2 * dGamma_dt) / self.k1))
        rheopectic_force = self.c_rh * dGamma_dt * (self.s1 ** 2) * (Lambda)
        #dls_dt = 1/self.cs * (-(self.ks * ls) + active_force[int(t/self.sim_dt)])
        dls_dt = self._get_dls_dt(ls,active_force[int(t/self.sim_dt)])
        d2lm_dt = 1/self.m*(-(rheopectic_force + self.c_rh_min * dlm_dt + self.c1 * dls_dt) - self.km*(np.sign(lm) * (lm**2))+self.kt*(np.sign(self.delta-lm)* ((self.delta-lm)**2))-active_force[int(t/self.sim_dt)] - self.F0)
        #d2lm_dt = 1/self.m*(-self.c_rh/5*dlm_dt-self.km*(lm**2)+self.kt*((self.delta-lm)**2)-active_force[int(t/self.sim_dt)])
        return [dlm_dt,d2lm_dt,dLambda_dt,dls_dt]

    def muscle_response(self,X0,time_vector,active_force):
        solver = ode(self._solve_muscle_dynamics)
        solver.set_integrator('dopri5', atol = 1e-2,rtol=1e-2)
        solver.set_initial_value(X0)
        solver.set_f_params(active_force)
        solution = np.zeros((len(X0),len(time_vector)))
        solution[:,0] = X0
        for n in range(1,len(time_vector)):
            try:
                solver.integrate(time_vector[n])
                assert solver.successful()
                solution[:,n] = solver.y
            except AssertionError as e:
                solution = np.zeros((len(X0),len(time_vector)))

        estimated_force = self.kt * np.sign(self.delta - solution[0,:]) * (np.abs(self.delta - solution[0,:]) **2) * 0.1
        #estimated_force = self.kt * (np.sign(self.delta - solution[0,:]) * (np.abs(self.delta - solution[0,:]) ** (1)) + np.sign(self.delta - solution[0,:]) * (np.abs(self.delta - solution[0,:]) ** (1))) * 10
        estimated_force = estimated_force - estimated_force[0]
        estimated_force = np.clip(estimated_force,0,None)
        return estimated_force,solution

    def set_parameters(self,x):
        self.km = x[0]
        self.kt = x[1]
        self.m = x[2]
        self.cs = x[3]
        self.ks = x[4]
        self.ls0 = x[5]
        self.c_rh = x[6]
        self.c_rh_min = x[7]
        self.c1 = x[8]
        self.k1 = x[9]
        self.k2 = x[10]
        self.lambda0 = x[11]
        self.F0 = x[12]
        self.s1 = x[13]

    def get_parameters(self):
        x = [self.km, self.kt,self.m ,self.cs,self.ks,self.ls0,self.c_rh,self.c_rh_min,self.c1,self.k1,self.k2,self.lambda0,self.F0,self.s1]
        return x

    def get_initial_length(self):
        dls0 = self._get_dls_dt(self.ls0,0)

        #a = self.km + self.kt
        a = self.kt - self.km
        b = 2 * self.kt * self.delta
        c = self.kt*(self.delta**2) - self.c1 * dls0 - self.F0
        quadratic_discriminant = b**2 - 4 * a * c
        lm01 = (-b + quadratic_discriminant**(1/2)) / (2 * a)
        lm02 = (-b - quadratic_discriminant**(1/2)) / (2 * a)
        if((lm01 > 0 and lm02 > 0) or (lm01 < 0 and lm02 < 0)):
            return False
        if (lm01 > 0):
            return lm01
        if (lm02 > 0):
            return lm02
        #return (self.kt * self.delta - self.c1 * dls0) / (self.kt + self.km)
        #return (self.kt * self.delta - self.F0) / (self.kt + self.km)
        #return (self.kt * self.delta - self.F0 - self.c2 * self.lambda0) / (self.kt + self.km)

    def get_X0(self):
        lm0 = self.get_initial_length()
        X0 = [lm0,0,self.lambda0,self.ls0]
        return X0
       
    @staticmethod
    def get_stiffness_ratio(x):
        km = x[0]
        kt = x[1]
        return kt/km
