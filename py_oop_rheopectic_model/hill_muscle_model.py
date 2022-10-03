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
    
    def _solve_muscle_dynamics(self,t,X,active_force):
        lm, dlm_dt =  X
        d2lm_dt = 1/self.m*(-self.c*dlm_dt-self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)])
        return [dlm_dt,d2lm_dt]

    def muscle_response(self,X0,time_vector,active_force):
        solver = ode(self._solve_muscle_dynamics)
        solver.set_integrator('dopri5', atol = 1e-2,rtol=1e-2)
        solver.set_initial_value(X0)
        solver.set_f_params(active_force)
        solution = np.zeros((len(X0),len(time_vector)))
        solution[:,0] = X0
        for n in range(1,len(time_vector)):
            solver.integrate(time_vector[n])
            assert solver.successful()
            solution[:,n] = solver.y
        estimated_force = np.sign(self.delta - solution[0,:]) * np.abs(self.delta - solution[0,:]) * 10
        estimated_force = estimated_force - estimated_force[0]
        return estimated_force,solution
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
    
    def set_parameter(self, parameter, value):
        old_parameter_value = getattr(self, parameter)
        setattr(self, parameter,value)
        return old_parameter_value

    def get_parameters(self):
        x = [self.km,self.kt,self.m,self.c]
        return x

    def get_damping_ratio(self,x):
        km = self.km
        kt = self.kt
        m = self.m
        c = self.c
        natural_frequency = np.sqrt((km+kt)/m)
        damping_ratio = c / (2 * m * natural_frequency)
        return damping_ratio
    
    @staticmethod
    def get_stiffness_ratio(x):
        km = x[0]
        kt = x[1]
        return kt/km

    def get_initial_length(self):
        return self.kt * self.delta / (self.kt + self.km)

    def get_X0(self):
        lm0 = self.get_initial_length()
        X0 = [lm0,0]
        return X0

class rheopectic_hill_muscle_model(hill_muscle_model):
    # TODO: OVERWRITE METHODS
    def __init__(self, km,kt,m,c_rh,c1,c_rh_min,k1,k2,A,B,C,D,lambda0,F0,delta,sim_dt):
        super().__init__(km = km,kt = kt,m = m,c=0, delta = delta, sim_dt = sim_dt)
        self.k1 = k1
        self.k2 = k2
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.lambda0 = lambda0
        self.c_rh = c_rh
        self.c_rh_min = c_rh_min
        self.c1 = c1
        self.F0 = F0

    def _solve_muscle_dynamics(self,t,X,active_force):
        lm, dlm_dt, Lambda = X
        dlm_dt_temp = dlm_dt
        dLambda_dt = -self.k1 * (np.abs(dlm_dt_temp)**self.A) * (Lambda**self.B) + self.k2 *(np.abs(dlm_dt_temp)**self.C) * (1-Lambda)**(self.D)
        d2lm_dt = 1/self.m*(-(self.c_rh * Lambda * dlm_dt_temp + self.c_rh_min * dlm_dt + self.c1 * Lambda) -self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)] - self.F0)
        return [dlm_dt,d2lm_dt,dLambda_dt]

    def set_parameters(self,x):
        self.c_rh = x[0]
        self.c_rh_min = x[1]
        self.c1 = x[2]
        self.k1 = x[3]
        self.k2 = x[4]
        self.C = x[5]
        self.D = x[6]
        self.lambda0 = x[7]
        self.F0 = x[8]

    def get_parameters(self):
        x = [self.c_rh,self.c_rh_min,self.c1,self.k1,self.k2,self.C,self.D,self.lambda0,self.F0]
        return x

    def get_initial_length(self):
        return (self.kt * self.delta - self.F0) / (self.kt + self.km)

    def get_X0(self):
        lm0 = self.get_initial_length()
        X0 = [lm0,0,self.lambda0]
        return X0



class modified_hill_muscle_model():
    def __init__(self,km,kt,m,c,c1,cs,ks,ls0,sg,F0,delta,sim_dt) -> None:
        self.km = km
        self.kt = kt
        self.m = m
        self.c = c
        self.delta = delta
        self.sim_dt = sim_dt
        self.cs = cs
        self.c1 = c1
        self.ks = ks
        self.ls0 = ls0
        self.F0 = F0
        self.sg = sg
    
    def _solve_muscle_dynamics(self,t,X,active_force):
        lm, dlm_dt, ls =  X
        dls_dt = 1/self.cs * (-self.ks * ls + active_force[int(t/self.sim_dt)])
        d2lm_dt = 1/self.m*(-self.c*dlm_dt-self.c1*dls_dt-self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)] - self.F0)
        return [dlm_dt,d2lm_dt,dls_dt]

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
            
        #estimated_force = self.kt * np.sign(self.delta - solution[0,:]) * (np.abs(self.delta - solution[0,:])) / 10 #**(1/2)) #/ 2000 
        estimated_force = self.kt * np.sign(self.delta - solution[0,:]) * (self.sg * np.abs(self.delta - solution[0,:]) ** (1/2)) 
        #estimated_force = self.kt * (np.sign(self.delta - solution[0,:]) * (np.abs(self.delta - solution[0,:]) ** (1)) + np.sign(self.delta - solution[0,:]) * (np.abs(self.delta - solution[0,:]) ** (1))) * 10
        estimated_force = estimated_force - estimated_force[0]
        estimated_force = np.clip(estimated_force,0,None)
        return estimated_force,solution

    def set_parameters(self,x):
        self.c = x[0]
        self.c1 = x[1]
        self.cs = x[2]
        self.ks = x[3]
        self.ls0 = x[4]
        self.F0 = x[5]
    
    def set_parameter(self, parameter, value):
        old_parameter_value = getattr(self, parameter)
        setattr(self, parameter,value)
        return old_parameter_value

    def get_parameters(self):
        #x = [self.km,self.kt,self.m,self.c,self.c1,self.cs,self.ks,self.ls0,self.F0]
        x = [self.c,self.c1,self.cs,self.ks,self.ls0,self.F0]
        return x

    def get_damping_ratio(self,x):
        km = self.km
        kt = self.kt
        m = self.m
        c = self.c
        natural_frequency = np.sqrt((km+kt)/m)
        damping_ratio = c / (2 * m * natural_frequency)
        return damping_ratio
    
    @staticmethod
    def get_stiffness_ratio(x):
        km = x[0]
        kt = x[1]
        return kt/km

    def get_initial_length(self):
        return (self.kt * self.delta - self.F0) / (self.kt + self.km)

    def get_X0(self):
        lm0 = self.get_initial_length()
        X0 = [lm0,0,self.ls0]
        return X0

class rheopectic_modified_hill_muscle_model(modified_hill_muscle_model):
 # TODO: OVERWRITE METHODS
    def __init__(self, km,kt,m,cs,ks,ls0,c_rh,c_rh_min,c1,k1,k2,A,B,C,D,lambda0,F0,sg,delta,sim_dt):
        super().__init__(km = km,kt = kt,m = m,c=0,c1 = c1,cs = cs, ks = ks,ls0 = ls0,F0=F0,sg=sg, delta = delta, sim_dt = sim_dt)
        self.k1 = k1
        self.k2 = k2
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.lambda0 = lambda0
        self.c_rh = c_rh
        self.c_rh_min = c_rh_min
        self.sg = sg

    @staticmethod
    def get_stiffness_ratio(x):
        km = x[6]
        kt = x[7]
        return kt/km

    def _get_dls_dt(self, current_ls, current_active_force):
        dls_dt = 1/self.cs * (-(self.ks * current_ls) + current_active_force)
        return dls_dt

    def _solve_muscle_dynamics(self,t,X,active_force):
        lm, dlm_dt, Lambda,ls = X
        Lambda = np.clip(Lambda,0,1)
        dlm_dt_temp = np.max([0,dlm_dt])
        dLambda_dt = -self.k1 * (dlm_dt_temp**self.A) * (Lambda**self.B) + self.k2 *(dlm_dt_temp**self.C) * (1-Lambda)**(self.D)
        #dls_dt = 1/self.cs * (-(self.ks * ls) + active_force[int(t/self.sim_dt)])
        dls_dt = self._get_dls_dt(ls,active_force[int(t/self.sim_dt)])
        d2lm_dt = 1/self.m*(-(self.c_rh * Lambda * dlm_dt + self.c_rh_min * dlm_dt + self.c1 * dls_dt) - self.km*(lm)+self.kt*((self.sg * (self.delta-lm))**(1/2))-active_force[int(t/self.sim_dt)] - self.F0)
        #d2lm_dt = 1/self.m*(-self.c_rh/5*dlm_dt-self.km*(lm**2)+self.kt*((self.delta-lm)**2)-active_force[int(t/self.sim_dt)])
        return [dlm_dt,d2lm_dt,dLambda_dt,dls_dt]

    def set_parameters(self,x):
        self.k1 = x[0]
        self.k2 = x[1]
        self.c_rh = x[2]
        self.c_rh_min = x[3]
        self.ls0 = x[4]
        self.c1 = x[5]
        self.cs = x[6]
        self.ks = x[7]
        self.lambda0 = x[8]
        self.A = x[9]
        self.B = x[10]
        self.C = x[11]
        self.D = x[12]
        self.F0 = x[13]
        self.km = x[14]
        self.kt = x[15]
        self.sg = x[16]
        
        '''
        self.c1 = x[0]
        self.cs = x[1]
        self.ks = x[2]
        '''

    def get_parameters(self):
        x = [self.k1,self.k2,self.c_rh,self.c_rh_min,self.ls0,self.c1,self.cs,self.ks,self.lambda0,self.A,self.B,self.C,self.D,self.F0,self.km,self.kt,self.sg]
        #x = [self.c1,self.cs, self.ks]
        return x

    def get_initial_length(self):
        dls0 = self._get_dls_dt(self.ls0,0)

        #For sqrt
        a = self.km**2
        b = -2 * self.km * self.c1 * dls0 + self.kt**2 + self.sg
        c = (dls0**2) * (self.c1**2) - self.sg * self.delta * self.kt**2 

        #For 2nd root
        #a = self.km + self.kt
        #b = 2 * self.kt * self.delta
        #c = -self.kt*(self.delta**2) - self.c1 * dls0
        quadratic_discriminant = b**2 - 4 * a * c
        lm01 = (-b + quadratic_discriminant**(1/2)) / (2 * a)
        lm02 = (-b - quadratic_discriminant**(1/2)) / (2 * a)
        if((lm01 > 0 and lm02 > 0) or (lm01 < 0 and lm02 < 0)):
            return False
        if (lm01 > 0):
            return lm01
        if (lm02 > 0):
            return lm02
        
        return (self.kt * self.delta - self.c1 * dls0) / (self.kt + self.km)
        #return (self.kt * self.delta - self.F0) / (self.kt + self.km)
        #return (self.kt * self.delta - self.F0 - self.c2 * self.lambda0) / (self.kt + self.km)

    def get_X0(self):
        lm0 = self.get_initial_length()
        X0 = [lm0,0,self.lambda0,self.ls0]
        return X0
