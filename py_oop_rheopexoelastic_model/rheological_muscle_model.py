from scipy.integrate import ode
import numpy as np
import time
from matplotlib import pyplot as plt 


cc_rh = []
Lambddd = []

class RheopecticMuscle():
    def __init__(self, km,kt,m,c_rh_min,c_rh_max,tr,Beta,k1,k2,lambda0,A,B, G0,delta,sim_dt):
        self.k1 = k1
        self.k2 = k2
        self.m = m
        self.km = km
        self.kt = kt
        self.lambda0 = lambda0
        self.c_rh_min = c_rh_min
        self.c_rh_max = c_rh_max
        self.tr = tr
        self.Beta = Beta
        self.K = np.power(self.c_rh_min/self.c_rh_max,1/self.Beta,dtype=float) - 1.0
        self.delta = delta
        self.sim_dt = sim_dt
        self.A = A
        self.B = B
        self.G0 = G0
        self.rh0 = 0
        self.rh_lambda_b = 0.54
        self.rh_lambda_c = 0.1

    def _solve_muscle_dynamics(self,t,X,active_force,X0):
        lm, dlm_dt,Lambda,rh = X
        #lm, dlm_dt = X
        #Lambda = np.clip(Lambda,0,1)
        
        '''
        rh_bias = 0
        lm_a = 1/((((X0[0] - rh_bias) - 3 * (0 + rh_bias)) / 2) * ((-(X0[0] - rh_bias) - (0+rh_bias))/2))
        lm = lm_a * (lm - (0+rh_bias)) * (lm - (X0[0]-rh_bias))
        '''

        #Lambda = 1
        
        if Lambda < 0:
            Lambda = 0.0
        elif Lambda > 1:
            Lambda = 1.0
        if lm < 0:
            lm = 0.0
        elif lm > X0[0]:
            lm = X0[0]
        
        #Lambda = 0
        #lm = np.clip(lm,0,X0[0])
        #lm = np.max([0,np.min([lm,X0[0]])])
        if((lm <= 0 and dlm_dt < 0) or (lm >= X0[0] and dlm_dt > 0)):
            dlm_dt = 0
        #dGamma_dt = np.max([0,dlm_dt]) / self.s1
        dGamma_dt = dlm_dt #/ 1 #self.s1
        dLambda_dt = - self.k1 * np.sign(dGamma_dt) * np.power(np.abs(dGamma_dt),self.A) * Lambda + self.k2 * np.sign(dGamma_dt) * np.power(np.abs(dGamma_dt),self.B) * (1-Lambda)
        #dLambda_dt = 0
        #c_rh = (self.c_rh_max - self.c_rh_min) * np.power((1 + self.K * Lambda),self.Beta,dtype=float) + self.c_rh_min
        c_rh = self.get_c_rh(Lambda)
        #cc_rh.append(c_rh)
        #X = np.power(self.c_rh_max / self.c_rh_min,0.5)
        #c_rh = self.c_rh_max * np.power((1 - (1 - X) * Lambda),-2)
        #c_rh_bias = 0
        #c_rh_a = 1/((((self.c_rh_max - c_rh_bias) - 3 * (self.c_rh_min + c_rh_bias)) / 2) * ((-(self.c_rh_max - c_rh_bias) - (self.c_rh_min+c_rh_bias))/2)) 
        #c_rh = 0.001 * c_rh_a * (c_rh - (self.c_rh_min+c_rh_bias)) * (c_rh - (self.c_rh_max-c_rh_bias))
        #c_rh = np.min([self.c_rh_max,np.max([c_rh,self.c_rh_min])])
        #c_rh = np.max([c_rh,1])
        #print(np.power((1 + self.K * Lambda),self.Beta))
        #c_rh = self.c_rh_min
        #c_rh = self.c_rh_min
        
        if((Lambda <= 0 and dLambda_dt <0) or (Lambda >= 1 and dLambda_dt > 0)):
            dLambda_dt = 0.0
        #rheopectic_force = c_rh * dGamma_dt #+ self.rh0#+ self.rh0 * dGamma_dt#+ self.rh0 * dGamma_dt
        
        #d2lm_dt = 1/self.m*(-rheopectic_force - self.km*(np.sign(lm) * np.square(lm)) +self.kt*(np.sign(self.delta-lm)* (np.square(self.delta - lm)))-self.act1*active_force[int(t/self.sim_dt)])
        #d2lm_dt = 1/self.m*(-rheopectic_force - self.km * (np.sign(lm) * np.square(lm)) + self.kt*(np.sign(self.delta-lm)* (np.square(self.delta - lm)))-np.max([0,active_force[int(t/self.sim_dt)]]))

        #if(np.isnan(lm)):
        #    exit()


        if(self.K == 0 or self.Beta == 0):
            f_visco = c_rh*dlm_dt
        else:
            #f_visco = rh
            # based on: https://www.sciencedirect.com/science/article/abs/pii/S0001868608001735?via%3Dihub
            #f_visco = rh + self.c_rh_min*dlm_dt + c_rh*dlm_dt
            f_visco = rh + self.c_rh_min*dlm_dt + Lambda*self.c_rh_min*dlm_dt
        #d2lm_dt = 1/self.m*(-rh - self.km * (np.sign(lm) * lm) + self.kt*(np.sign(self.delta-lm)* (self.delta - lm))-np.max([0,active_force[int(t/self.sim_dt)]]))
        d2lm_dt = 1/self.m*(-f_visco - self.km * (np.sign(lm) * lm) + self.kt*(np.sign(self.delta-lm)* (self.delta - lm))-np.max([0,active_force[int(t/self.sim_dt)]]))
        #d2lm_dt = 1/self.m*(-(rheopectic_force) - self.km*lm +self.kt*(self.delta - lm)-active_force[int(t/self.sim_dt)])
        #d2lm_dt = 1/self.m*(-rheopectic_force - self.km*lm - self.kt*(lm)-active_force[int(t/self.sim_dt)])

        #drh_dt = (Lambda + self.c_rh_min/c_rh) * dlm_dt * self.G0 + (self.c_rh_min * d2lm_dt) - rh*self.G0/c_rh

        #drh_dt = self.G0 * (dlm_dt + d2lm_dt*self.tr - rh/c_rh)
        #drh_dt = self.G0 * (dlm_dt + d2lm_dt*self.tr - rh/c_rh)
        #drh_dt = self.G0 * ((Lambda + (self.tr * self.G0)/c_rh)*dlm_dt + d2lm_dt*self.tr - rh/c_rh)
        # VER1 OK:
        
        #G = self.G0 * (1-Lambda)
        #G = self.G0 * np.exp((-(Lambda-self.rh_lambda_b)**2)/(2*self.rh_lambda_c**2))
        #G = self.G0 * (1-Lambda) #* (1-Lambda)
        G = self.G0 * (Lambda)
        drh_dt = G * (dlm_dt + d2lm_dt*self.tr - rh/c_rh)
        # VER2 OK:
        #drh_dt = self.G0 * (dlm_dt + d2lm_dt*self.tr - rh/c_rh)
        # VER3 OK:
        #drh_dt = self.G0 * ((Lambda + self.c_rh_max/c_rh)*dlm_dt + d2lm_dt*self.c_rh_max/self.G0 - 0.5 * rh/c_rh)
        #if drh_dt < -100:
        #    drh_dt = -100
        #elif drh_dt > 100:
        #    drh_dt = 100
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
                st = time.time()
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
        #plt.figure()
        #plt.plot(cc_rh)
        #plt.plot(Lambddd)
        #plt.show()
        #exit()
        #estimated_force = np.clip(estimated_force,0,np.inf)
        return estimated_force,solution


    def _integrate_rk4(self,f, x0, time_vector, dt,active_force):
        t = time_vector
        n = len(t)
        solution = np.zeros((n, len(x0)))
        solution[0] = x0
        for i in range(n - 1):
            try: 
                k1 = f(t[i], solution[i],active_force,x0)
                k2 = f(t[i] + dt/2, solution[i] + dt/2 * k1,active_force,x0)
                k3 = f(t[i] + dt/2, solution[i] + dt/2 * k2,active_force,x0)
                k4 = f(t[i] + dt, solution[i] + dt * k3,active_force,x0)
                solution[i+1] = solution[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            except:
                solution = np.zeros((len(time_vector),(len(x0))))
                break
        solution = solution.transpose()
        estimated_force = self.kt * np.sign(self.delta - solution[0,:]) * (np.abs(self.delta - solution[0,:]))
        estimated_force = estimated_force - estimated_force[0]
        return estimated_force,solution
    
    def muscle_response_rk4(self,X0,time_vector,active_force):
        x0 = X0
        dt = self.sim_dt
        estimated_force,solution = self._integrate_rk4(self._solve_muscle_dynamics, x0, time_vector, dt,active_force)
        return estimated_force,solution

    def set_parameters(self,x):
        
        self.km = x[0]
        self.kt = x[1]
        self.m = x[2]
        self.c_rh_min = x[3]
        self.c_rh_max = x[4]
        self.tr = x[5]
        #self.K = x[6]
        self.Beta = x[6]
        self.k1 = x[7]
        self.k2 = x[8]
        self.A = x[9]
        self.B = x[10]
        self.G0 = x[11]
        self.lambda0 = x[12]
        '''
        self.rh_lambda_b = x[0]
        self.rh_lambda_c = x[1]
        '''

    def get_parameters(self):
        x = [self.km, self.kt,self.m ,self.c_rh_min,self.c_rh_max,self.tr,self.Beta,self.k1,self.k2,self.A,self.B,self.G0,self.lambda0]
        #x = [self.rh_lambda_b,self.rh_lambda_c]
        return x

    def get_initial_length(self):
        return ((self.kt * self.delta - self.rh0) / (self.km + self.kt))


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

    def get_c_rh(self,Lambda):
        #return self.c_rh_max * np.power((1 + self.K * Lambda),self.Beta,dtype=float)
        return np.max([self.c_rh_min,np.min([self.c_rh_max,self.c_rh_max * np.power((1 + self.K * Lambda),self.Beta,dtype=float)])])
    
        #return (self.c_rh_max - self.c_rh_min) * np.power((1 + self.K * Lambda),self.Beta,dtype=float) + self.c_rh_min

    def get_X0(self):
        lm0 = self.get_initial_length()
        X0 = [lm0,0,self.lambda0,self.rh0]
        #X0 = [lm0,0]
        return X0

    @staticmethod
    def get_stiffness_ratio(x):
        km = x[0]
        kt = x[1]
        return kt/km
    

    def get_initial_force(self,x):
        return self.delta * x[1] - 0.2
