from scipy.integrate import ode
import numpy as np

class hill_muscle_model:
    def __init__(self,km,kt,m,c,F_K,F_delta,delta,sim_dt) -> None:
    #def __init__(self,km,kt,m,c,delta,sim_dt) -> None:
        self.km = km
        self.kt = kt
        self.m = m
        self.c = c
        self.F_K = F_K
        self.F_delta = F_delta
        self.delta = delta
        self.sim_dt = sim_dt

    def _solve_muscle_dynamics(self,t,X,active_force):
        lm, dlm_dt = X
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
            #solution[2,n] = np.max([0.0, np.min([1.0, solution[2,n]])])
        estimated_force = np.sign(self.delta - solution[0,:]) * np.abs(self.delta - solution[0,:]) * 10
        #estimated_force = self.kt * np.sign(self.delta - solution[0,:]) * np.abs(self.delta - solution[0,:])
        estimated_force = estimated_force - estimated_force[0]
        #estimated_force = 1 / (1 + np.exp(-self.F_K*(estimated_force-self.F_delta)))
        
        #estimated_force = estimated_force - estimated_force[0]
        return estimated_force,solution

    '''
    def muscle_response(self,X0,time_vector,active_force,F_K = 0,F_delta = 0):
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
        estimated_force = self.kt * np.sign(self.delta - solution[0,:]) * np.abs(self.delta - solution[0,:])
        #estimated_force = (1 / (1 + np.exp(-F_K*(estimated_force-F_delta))))
        estimated_force = estimated_force - estimated_force[0]
        return estimated_force
    '''
    def set_parameters(self,km,kt,m,c,F_K,F_delta):
        self.km = km
        self.kt = kt
        self.m = m
        self.c = c
        self.F_K = F_K
        self.F_delta = F_delta

    def set_parameters(self,x):
        self.km = x[0]
        self.kt = x[1]
        self.m = x[2]
        self.c = x[3]
        self.F_K = x[4]
        self.F_delta = x[5]
    
    def set_parameter(self, parameter, value):
        old_parameter_value = getattr(self, parameter)
        setattr(self, parameter,value)
        return old_parameter_value

    def get_parameters(self):
        x = [self.km,self.kt,self.m,self.c,self.F_K,self.F_delta]
        #x = [self.km,self.kt,self.m,self.c]
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
    
    @staticmethod
    def get_stiffness_ratio(x):
        km = x[0]
        kt = x[1]
        return kt/km

    def get_initial_length(self):
        return self.kt * self.delta / (self.km + self.kt)

    def get_X0(self):
        lm0 = self.get_initial_length()
        X0 = [lm0,0]
        return X0

class rheopectic_hill_muscle_model(hill_muscle_model):
    # TODO: OVERWRITE METHODS
    def __init__(self, km,kt,m,c0,min_c,max_c,k1,k2,A,B,C,D,lambda0,delta,sim_dt):
        super().__init__(km = km,kt = kt,m = m,c=0, F_K = 0, F_delta = 0, delta = delta, sim_dt = sim_dt)
        self.k1 = k1
        self.k2 = k2
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.lambda0 = lambda0
        self.c0 = c0
        self.min_c = min_c
        self.max_c = max_c

    def _solve_muscle_dynamics(self,t,X,active_force):
        lm, dlm_dt, Lambda = X
        #Lambda = np.max([0.0, np.min([1.0, Lambda])])
        dLambda_dt = -self.k1*np.sign(dlm_dt)*(np.abs(dlm_dt)**(self.A))*(Lambda**(self.B))+self.k2*np.sign(dlm_dt)*(np.abs(dlm_dt)**(self.C))*((1-Lambda)**(self.D))
        if(Lambda >= 1.0 and dLambda_dt > 0.0):
            Lambda = 1.0
            dLambda_dt = 0.0
        elif(Lambda <= 0.0 and dLambda_dt < 0.0):
            Lambda = 0.0
            dLambda_dt = 0.0
        #print(np.abs(dlm_dt))
        K = 1 - np.sqrt(self.max_c/self.min_c)
        c = self.max_c/((1 - K * Lambda)**2) + self.c0
        #print(c)

        #Fd = -c*dlm_dt
        #Fd = Lambda * 1 + Lambda * self.max_c * dlm_dt + self.min_c*dlm_dt
        d2lm_dt = 1/self.m*(-c*dlm_dt-self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)])
        return [dlm_dt,d2lm_dt,dLambda_dt]

    def set_parameters(self,min_c,k1,C,D,c0,lambda0):
        self.min_c = min_c
        self.k1 = k1
        self.C = C
        self.D = D
        self.c0 = c0
        self.lambda0 = lambda0

    def set_parameters(self,x):
        self.min_c = x[0]
        self.k1 = x[1]
        self.C = x[2]
        self.D = x[3]
        self.c0 = x[4]
        self.lambda0 = x[5]
        self.max_c = x[6]

    def get_parameters(self):
        x = [self.min_c,self.k1,self.C,self.D,self.c0,self.lambda0,self.max_c]
        return x

    def get_X0(self):
        lm0 = self.get_initial_length()
        X0 = [lm0,0,self.lambda0]
        return X0


