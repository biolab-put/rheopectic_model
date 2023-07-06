from scipy.integrate import ode
import numpy as np
import time
from matplotlib import pyplot as plt 


rheo_force = []

class RheopecticMuscle():
    def __init__(self, km,kt,m,c_rh_min,tr,K,Beta,k1,k2,lambda0,A,B, G0,delta,sim_dt):
        self.k1 = k1
        self.k2 = k2
        self.m = m
        self.km = km
        self.kt = kt
        self.lambda0 = lambda0
        self.c_rh_min = c_rh_min
        self.tr = tr
        self.K = K
        self.Beta = Beta
        self.delta = delta
        self.sim_dt = sim_dt
        self.A = A
        self.B = B
        self.G0 = G0

    def _solve_muscle_dynamics(self,t,X,active_force,X0):
        lm, dlm_dt,Lambda,rh = X
        #lm, dlm_dt = X
        Lambda = np.clip(Lambda,0,1)
        #Lambda = 0
        lm = np.clip(lm,0,X0[0])
        #lm = np.max([0,np.min([lm,X0[0]])])
        if((lm <= 0 and dlm_dt < 0) or (lm >= X0[0] and dlm_dt > 0)):
            dlm_dt = 0
        #dGamma_dt = np.max([0,dlm_dt]) / self.s1
        dGamma_dt = dlm_dt #/ 1 #self.s1
        dLambda_dt = - self.k1 * np.sign(dGamma_dt) * np.power(np.abs(dGamma_dt),self.A) * Lambda + self.k2 * np.sign(dGamma_dt) * np.power(np.abs(dGamma_dt),self.B) * (1-Lambda)
        
        c_rh = self.c_rh_min * np.power((1 + self.K * Lambda),self.Beta)
        #c_rh = self.c_rh_min
        #c_rh = self.c_rh_min
        if((Lambda <= 0 and dLambda_dt <0) or (Lambda >= 1 and dLambda_dt > 0)):
            dLambda_dt = 0
        #rheopectic_force = c_rh * dGamma_dt #+ self.rh0#+ self.rh0 * dGamma_dt#+ self.rh0 * dGamma_dt
        rheopectic_force = rh
        #d2lm_dt = 1/self.m*(-rheopectic_force - self.km*(np.sign(lm) * np.square(lm)) +self.kt*(np.sign(self.delta-lm)* (np.square(self.delta - lm)))-self.act1*active_force[int(t/self.sim_dt)])
        #d2lm_dt = 1/self.m*(-rheopectic_force - self.km * (np.sign(lm) * np.square(lm)) + self.kt*(np.sign(self.delta-lm)* (np.square(self.delta - lm)))-np.max([0,active_force[int(t/self.sim_dt)]]))

        d2lm_dt = 1/self.m*(-rheopectic_force - self.km * (np.sign(lm) * lm) + self.kt*(np.sign(self.delta-lm)* (self.delta - lm))-np.max([0,active_force[int(t/self.sim_dt)]]))
        #d2lm_dt = 1/self.m*(-(rheopectic_force) - self.km*lm +self.kt*(self.delta - lm)-active_force[int(t/self.sim_dt)])
        #d2lm_dt = 1/self.m*(-rheopectic_force - self.km*lm - self.kt*(lm)-active_force[int(t/self.sim_dt)])

        #drh_dt = (Lambda + self.c_rh_min/c_rh) * dlm_dt * self.G0 + (self.c_rh_min * d2lm_dt) - rh*self.G0/c_rh

        drh_dt = self.G0 * (dlm_dt + d2lm_dt*self.tr - rh/c_rh)
        #rheo_force.append(lm)
        #rheo_force.append(np.max([0,active_force[int(t/self.sim_dt)]]))
        #return [dlm_dt,d2lm_dt]
        return [dlm_dt,d2lm_dt,dLambda_dt,drh_dt]

    def muscle_response(self,X0,time_vector,active_force):
        solver = ode(self._solve_muscle_dynamics)
        solver.set_integrator('dopri5', atol = 1e-4,rtol=1e-4)
        solver.set_initial_value(X0)
        solver.set_f_params(active_force,X0)
        solution = np.zeros((len(X0),len(time_vector)))
        solution[:,0] = X0
        for n in range(1,len(time_vector)):
            try:   
                solver.integrate(time_vector[n])
                assert solver.successful()
                solution[:,n] = solver.y
            except AssertionError as e:
                solution = np.zeros((len(X0),len(time_vector)))
                break
        # Muscle length signal does not start with 0, but do not change it before calculating estimated force
        # because it should take into account delta - lm, it should depend as a non linear output from delta value
        #estimated_force = self.kt * np.sign(self.delta - solution[0,:]) * (np.square(np.abs(self.delta - solution[0,:])))
        estimated_force = self.kt * np.sign(self.delta - solution[0,:]) * (np.abs(self.delta - solution[0,:]))
        #estimated_force = self.kt * (self.delta - solution[0,:])
        #estimated_force = self.kt * (-solution[0,:])
        estimated_force = estimated_force - estimated_force[0]
        #estimated_force = np.clip(estimated_force,0,np.inf)

        #plt.plot(rheo_force)
        #plt.show()
        return estimated_force,solution

    def set_parameters(self,x):
        self.km = x[0]
        self.kt = x[1]
        self.m = x[2]
        self.c_rh_min = x[3]
        self.tr = x[4]
        self.K = x[5]
        self.Beta = x[6]
        self.k1 = x[7]
        self.k2 = x[8]
        self.A = x[9]
        self.B = x[10]
        self.G0 = x[11]
        self.lambda0 = x[12]

    def get_parameters(self):
        #x = [self.km, self.kt,self.m ,self.c_rh_min]
        x = [self.km, self.kt,self.m ,self.c_rh_min,self.tr,self.K,self.Beta,self.k1,self.k2,self.A,self.B,self.G0,self.lambda0]
        #x = [self.km, self.kt,self.m ,self.c_rh_min]#,self.K,self.Beta,self.k1,self.k2,self.A,self.B,self.lambda0,self.delta]
        return x

    def get_initial_length(self):
        return (self.kt * self.delta / (self.km + self.kt))


        a = self.kt - self.km
        b = -2 * self.kt * self.delta
        c = self.kt*(np.square(self.delta))
        quadratic_discriminant = np.square(b) - 4 * a * c
        lm01 = (-b + np.sqrt(quadratic_discriminant)) / (2 * a)
        lm02 = (-b - np.sqrt(quadratic_discriminant)) / (2 * a)
        #if((lm01 > 0 and lm02 > 0) or (lm01 < 0 and lm02 < 0)):
        #    return False

        if(lm01 < 0 and lm02 < 0):
            return False
        if(lm01 < lm02):
            return lm01
        else:
            return lm02
        #if (lm01 > 0):
        #    return lm01
        #if (lm02 > 0):
        #    return lm02

    def get_X0(self):
        lm0 = self.get_initial_length()
        X0 = [lm0,0,self.lambda0,0]
        #X0 = [lm0,0]
        return X0

    @staticmethod
    def get_stiffness_ratio(x):
        km = x[0]
        kt = x[1]
        return kt/km
    

    def get_initial_force(self,x):
        return self.delta * x[1] - 0.2
