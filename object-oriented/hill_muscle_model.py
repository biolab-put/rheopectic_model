from scipy.integrate import ode
import numpy as np
from matplotlib import pyplot as plt 



class hill_muscle_model:
    def __init__(self,km,kt,m,c,delta,sim_dt) -> None:
    #def __init__(self,km,kt,m,c,delta,sim_dt) -> None:
        self.km = km
        self.kt = kt
        self.m = m
        self.c = c
        self.delta = delta
        self.sim_dt = sim_dt


    
    def _solve_muscle_dynamics(self,t,X,active_force):
        lm, dlm_dt =  X#,Fact = X #,Cn,F = X

        #T = 0.7

        #ind = np.min([int(t/self.sim_dt)+1,len(active_force)-1])
        #dx_dt = active_force[ind] - active_force[int(t/self.sim_dt)]

        #tau_c = 0.02
        #KM = 0.9
        #tau_1 = 0.03
        #tau_2 = 0.2

        #dCn_dt  = active_force[int(t/self.sim_dt)] - Cn/tau_c
        #dF_dt = 1 * Cn / (KM + Cn) - F/(tau_1 + tau_2*Cn/(KM+Cn))

        

        
        #dFact_dt = 1/T * (1 * (dx_dt/self.sim_dt + active_force[int(t/self.sim_dt)] + 1 * T * dx_dt/self.sim_dt) - Fact)
        #dFact_dt = 1/T * (1 * (dx_dt/self.sim_dt + active_force[int(t/self.sim_dt)]) - Fact)
        #if(Fact < 0):
        #    Fact = 0
        #dFact_dt = 1
        

        d2lm_dt = 1/self.m*(-self.c*dlm_dt-self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)])
        #test.append(dFact_dt)
        
        #d2lm_dt = 1/self.m*(-c*dlm_dt-self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)])
        #d2lm_dt = 1/self.m*(-self.c*dlm_dt-(np.sign(lm)*self.km*lm**2+self.km*lm)+(np.sign((self.delta-lm))*self.kt*(self.delta-lm)**2 + self.kt*(self.delta-lm))-active_force[int(t/self.sim_dt)])
        #d2lm_dt = 1/self.m*(-c*dlm_dt-(np.exp(self.km*lm)-1)+(np.exp(self.kt*(self.delta-lm))-1)-active_force[int(t/self.sim_dt)])
        return [dlm_dt,d2lm_dt]#,dFact_dt]#,dCn_dt,dF_dt]


    def _solve_force_dynamics(self,t,X,active_force):
        Fact = X
        T = 0.05
        dFact_dt = 1/T * (1 * active_force[int(t/self.sim_dt)] - Fact)
        return dFact_dt


    def muscle_response(self,X0,time_vector,active_force):

        #force_solver = ode(self._solve_force_dynamics)
        #force_solver.set_integrator('dopri5', atol = 1e-2,rtol=1e-2)

        #force_solver.set_initial_value([0])
        #force_solver.set_f_params(active_force)
        #Fact = np.zeros((1,len(time_vector)))
        #Fact[:,0] = 0
        #for n in range(1,len(time_vector)):
        #    force_solver.integrate(time_vector[n])
        #    assert force_solver.successful()
        #    Fact[:,n] = force_solver.y

        #plt.plot(Fact[0,:])
        #plt.show()
        #exit()
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
        #estimated_force = self.kt * np.sign(self.delta - solution[0,:]) * np.abs(self.delta - solution[0,:])**2 + self.kt * (self.delta - solution[0,:])
        #np.exp(self.kt*(self.delta-lm))-1
        #plt.figure(1)
        #plt.plot(test)
        #plt.figure(2)
        #plt.plot(solution[2,:])
        #plt.show()
 
        estimated_force = estimated_force - estimated_force[0]



        #x = np.linspace(0,10,1000)
        #a = 0.1
        
        #np.min(estimated_force)
        #plt.plot(np.minimum(np.maximum(x,a*x) + 1*a,1))
        #estimated_force = 1 / (1 + np.exp(-(self.F_K * estimated_force-self.F_delta)))
        #plt.plot(estimated_force)
        #plt.xlabel('Estimated force')
        #plt.ylabel('Transformed force')
        #plt.grid()
        #plt.show()
 
        #plt.plot(estimated_force)
        #plt.plot(solution[2,:])
        #plt.show()
        #exit()
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
        #self.F_K = F_K
        #self.F_delta = F_delta

    def set_parameters(self,x):
        self.km = x[0]
        self.kt = x[1]
        self.m = x[2]
        self.c = x[3]
        #self.F_K = x[4]
        #self.F_delta = x[5]
    
    def set_parameter(self, parameter, value):
        old_parameter_value = getattr(self, parameter)
        setattr(self, parameter,value)
        return old_parameter_value

    def get_parameters(self):
        #x = [self.km,self.kt,self.m,self.c,self.F_K,self.F_delta]
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
    
    def Lambda2viscosity(self,Lambda):
        #Lambda = 0
        max_c = self.max_c
        min_c = self.min_c
        if(self.C > self.A):
            max_c = self.min_c
            min_c = self.max_c

        K = 1 - np.sqrt(max_c/min_c)
        c = max_c/((1 - K * Lambda)**2) + self.c0
        #print(c)
        #exit()
        #else:
        #    K = 1 - np.sqrt(self.max_c/self.min_c)
        #    c = self.max_c/((1 - K * Lambda)**2) + self.c0
        #print(c)
        #exit()
        return c

    def _solve_muscle_dynamics(self,t,X,active_force):
        lm, dlm_dt, Lambda = X
        #dlm_dt_temp = np.max([0,dlm_dt])
        dlm_dt_temp = dlm_dt
        #dLambda_dt = -self.k1*np.sign(dlm_dt)*((np.abs(dlm_dt))**(self.A))*(Lambda**(self.B))+self.k2*np.sign(dlm_dt)*((np.abs(dlm_dt))**(self.C))*((1-Lambda)**(self.D))
        dLambda_dt = -self.k1 * (np.abs(dlm_dt_temp)**self.A) * (Lambda**self.B) + self.k2 *(np.abs(dlm_dt_temp)**self.C) * (1-Lambda)**(self.D)
        #if(self.k2*np.sign(dlm_dt)*(np.abs(dlm_dt)**(self.C))*((1-Lambda)**(self.D)) < 0):
        #    print('down', -self.k1*np.sign(dlm_dt)*(np.abs(dlm_dt)**(self.A))*(Lambda**(self.B)))
        #    print('up', self.k2*np.sign(dlm_dt)*(np.abs(dlm_dt)**(self.C))*((1-Lambda)**(self.D)))

        #dLambda_dt = -self.k1*np.sign(dlm_dt)*((1.0 + np.abs(dlm_dt))**(self.A) - 1)*(Lambda_temp**(self.B))+self.k2*np.sign(dlm_dt)*((1.0 + np.abs(dlm_dt))**(self.C)-1)*((1-Lambda_temp)**(self.D))
        #dLambda_dt = -self.k1*np.sign(dlm_dt)*((np.abs(dlm_dt))**(self.A))*(Lambda_temp**(self.B))+self.k2*np.sign(dlm_dt)*((np.abs(dlm_dt))**(self.C))*((1-Lambda_temp)**(self.D))
        #dLambda_dt = -self.k1*np.sign(dlm_dt)*((np.abs(dlm_dt))**(self.A))*(Lambda_temp**(self.B))+self.k2*np.sign(dlm_dt)*((np.abs(dlm_dt))**(self.C))*((1-Lambda_temp)**(self.D))
        #print(np.abs(dlm_dt))
        
        
        #K = 1 - np.sqrt(self.min_c/self.max_c)
        #c = self.min_c/((1 - K * Lambda)**2) + self.c0
        #print(c)

        #Fd = -c*dlm_dt
        #Fd = Lambda * 1 + Lambda * self.max_c * dlm_dt + self.min_c*dlm_dt

        #c = self.Lambda2viscosity(Lambda)
        #d2lm_dt = 1/self.m*(-c*dlm_dt-self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)])

        ####TODO zmieniona dlambda i rownanie na sile tlumika zakomentowane w d2lm_dt i cc0,c0 i w grid_search.py

        #d2lm_dt = 1/self.m*(-(self.c0 * Lambda * dlm_dt)-self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)])
        d2lm_dt = 1/self.m*(-(self.c_rh * Lambda * dlm_dt_temp + self.c_rh_min * dlm_dt + self.c1 * Lambda) -self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)] - self.F0)
        #d2lm_dt = 1/self.m*(-c*dlm_dt-(np.sign(lm)*self.km*lm**2+self.km*lm)+(np.sign((self.delta-lm))*self.kt*(self.delta-lm)**2 + self.kt*(self.delta-lm))-active_force[int(t/self.sim_dt)])
        #d2lm_dt = 1/self.m*(-c*dlm_dt-(np.exp(self.km*lm)-1)+(np.exp(self.kt*(self.delta-lm))-1)-active_force[int(t/self.sim_dt)])
        return [dlm_dt,d2lm_dt,dLambda_dt]

    def set_parameters(self,c_rh,k1,k2,C,D,c0,lambda0):
        pass
        '''
        self.c_rh = c_rh
        self.k1 = k1
        self.k2 = k2
        self.C = C
        self.D = D
        self.c0 = c0
        self.lambda0 = lambda0
        '''

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
    def __init__(self,km,kt,m,c,c1,cs,ks,ls0,F0,delta,sim_dt) -> None:
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
        #self.test = []
        

    def _solve_muscle_dynamics(self,t,X,active_force):
        lm, dlm_dt, ls =  X#,Fact = X #,Cn,F = X
        #T = 0.7
        #ind = np.min([int(t/self.sim_dt)+1,len(active_force)-1])
        dls_dt = 1/self.cs * (-self.ks * ls + active_force[int(t/self.sim_dt)])
        #p_act = self.ks * ls + self.cs * dls_dt #- active_force[int(t/self.sim_dt)]
        #d2lm_dt = 1/self.m*(-self.c*dls_dt-self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)])
        #self.c1 = 0
        #self.F0 = 0
        d2lm_dt = 1/self.m*(-self.c*dlm_dt-self.c1*dls_dt-self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)] - self.F0)
        #self.test.append(dls_dt)
        return [dlm_dt,d2lm_dt,dls_dt]

    def sarkomer_dynamics():
        #ls = k*p_act
        #ls + c * dls_dt
        #k*p_act + c * dls_dt
        pass
    def muscle_response(self,X0,time_vector,active_force):
        #solution = solve_ivp(fun = self._solve_muscle_dynamics, t_span = (time_vector[0],time_vector[-1]), y0 = X0, method = 'RK45',t_eval = time_vector,rtol=1e90,atol=1e90,max_step=1*self.sim_dt,first_step = 1*self.sim_dt,args = (active_force,))
        #print(solution)

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
            
        estimated_force = np.sign(self.delta - solution[0,:]) * np.abs(self.delta - solution[0,:]) * 10
        estimated_force = estimated_force - estimated_force[0]
        estimated_force = np.clip(estimated_force,0,None)
        #estimated_force = np.sign(self.delta - solution.y[0,:]) * np.abs(self.delta - solution.y[0,:]) * 10
        #estimated_force = estimated_force - estimated_force[0]
        #return estimated_force,solution.y
        return estimated_force,solution

    def set_parameters(self,km,kt,m,c,c1,cs,ks,F_K,F_delta):
        self.km = km
        self.kt = kt
        self.m = m
        self.c = c
        self.c1 = c1
        self.cs = cs
        self.ks = ks
        

    def set_parameters(self,x):
        #self.km = x[0]
        #self.kt = x[1]
        #self.m = x[2]
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
    def __init__(self, km,kt,m,cs,ks,ls0,c_rh,c_rh_min,c1,k1,k2,A,B,C,D,lambda0,F0,delta,sim_dt):
        super().__init__(km = km,kt = kt,m = m,c=0,c1 = c1,cs = cs, ks = ks,ls0 = ls0,F0=F0, delta = delta, sim_dt = sim_dt)
        self.k1 = k1
        self.k2 = k2
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.lambda0 = lambda0
        self.c_rh = c_rh
        self.c_rh_min = c_rh_min
        #self.c2 = c2
        #self.test = []
        #self.min_c = min_c
        #self.max_c = max_c
    
    '''
    def Lambda2viscosity(self,Lambda):
        max_c = self.max_c
        min_c = self.min_c
        if(self.C > self.A):
            max_c = self.min_c
            min_c = self.max_c

        K = 1 - np.sqrt(max_c/min_c)
        c = max_c/((1 - K * Lambda)**2) + self.c0
        return c
    '''

    @staticmethod
    def get_stiffness_ratio(x):
        km = x[6]
        kt = x[7]
        return kt/km

    def _solve_muscle_dynamics(self,t,X,active_force):
        lm, dlm_dt, Lambda,ls = X
        Lambda = np.clip(Lambda,0,1)
        #Lambda_temp = Lambda
        #dlm_dt_temp = np.max([0,dlm_dt])
        dlm_dt_temp = np.max([0,dlm_dt])
        
        #dLambda_dt = -self.k1*np.sign(dlm_dt)*((np.abs(dlm_dt))**(self.A))*(Lambda**(self.B))+self.k2*np.sign(dlm_dt)*((np.abs(dlm_dt))**(self.C))*((1-Lambda)**(self.D))
       # with np.errstate(invalid='raise'):
       #     try:
        dLambda_dt = -self.k1 * (dlm_dt_temp**self.A) * (Lambda**self.B) + self.k2 *(dlm_dt_temp**self.C) * (1-Lambda)**(self.D)
       #     except FloatingPointError:
       #         print('ERR')
       #         print('1', -self.k1 * (np.abs(dlm_dt_temp)**self.A) * (Lambda**self.B))
       #         print('2', self.k2, ' : ', (np.abs(dlm_dt_temp)**self.C))#, ' : ', (1-Lambda)**(self.D))
       #         print(Lambda)
       #         print('2.5', (1-Lambda))
       #         print('3', (1-Lambda)**(self.D))
                #print('2', self.k2 *(np.abs(dlm_dt_temp)**self.C) * (1-Lambda)**(self.D))
        #if(Lambda >= 0.97 and dLambda_dt > 0.0):
        #    Lambda = 1.0
        #    dLambda_dt = 0.0
        #elif(Lambda <= 0.01 and dLambda_dt < 0.0):
        #    Lambda = 0.0
        #    dLambda_dt = 0.0
        dls_dt = 1/self.cs * (-(self.ks * ls) + active_force[int(t/self.sim_dt)])
        #self.test.append(active_force[int(t/self.sim_dt)])
        #dls_dt = np.max([0,dls_dt])
        #print('a', dlm_dt_temp + self.c1 * dls_dt)
        #print('b', self.c_rh_min * np.abs(dlm_dt_temp))
        #print('c', self.c_rh * Lambda * np.abs(dlm_dt_temp))
        #+ self.c1 * dls_dt
        #K = 1 - np.sqrt(self.c_rh_min/self.c_rh_max)
        #c = self.c_rh_min / (1 - K * Lambda)**2
        #d2lm_dt = 1/self.m*(-(c * dlm_dt_temp + self.c0 * dlm_dt) -self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)] - self.F0)
        d2lm_dt = 1/self.m*(-(self.c_rh * Lambda * dlm_dt + self.c_rh_min * dlm_dt + self.c1 * dls_dt) -self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)] - self.F0)
        #d2lm_dt = 1/self.m*(-(self.c_rh * Lambda + self.c_rh_min) * dlm_dt - self.c1 * dls_dt -self.km*lm+self.kt*(self.delta-lm)-active_force[int(t/self.sim_dt)])
        #d2lm_dt = 1/self.m*(-c*dlm_dt-(np.sign(lm)*self.km*lm**2+self.km*lm)+(np.sign((self.delta-lm))*self.kt*(self.delta-lm)**2 + self.kt*(self.delta-lm))-active_force[int(t/self.sim_dt)])
        #d2lm_dt = 1/self.m*(-c*dlm_dt-(np.exp(self.km*lm)-1)+(np.exp(self.kt*(self.delta-lm))-1)-active_force[int(t/self.sim_dt)])
        return [dlm_dt,d2lm_dt,dLambda_dt,dls_dt]

    '''
    def set_parameters(self,min_c,k1,k2,C,D,c0,lambda0):
        self.min_c = min_c
        self.k1 = k1
        self.k2 = k2
        self.C = C
        self.D = D
        self.c0 = c0
        self.lambda0 = lambda0
    '''

    def set_parameters(self,x):
        #self.k1 = x[0]
        #self.k2 = x[1]
        #self.C = x[2]
        #self.D = x[3]
        #self.c0 = x[4]
        #self.lambda0 = x[5]
        #self.cs = x[6]
        #self.ks = x[7]
        #self.ls0 = x[8]
        #self.k1 = x[0]
        #self.c_rh = x[1]
        #self.ls0 = x[2]
        #self.A = x[3]
        self.km = x[0]
        self.kt = x[1]
        self.c1 = x[2]
        '''
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
        '''
        # stiffness ratio

    def get_parameters(self):
        #x = [self.min_c,self.k1,self.k2,self.C,self.D,self.c0,self.lambda0,self.max_c]
        #x = [self.k1,self.k2,self.C,self.D,self.c0,self.lambda0,self.cs,self.ks,self.ls0]
        #x = [self.k1,self.k2,self.C,self.D,self.c0,self.lambda0,self.cs,self.ks,self.ls0]
        #x = [self.k1,self.c_rh,self.ls0, self.A]
        #x = [self.k1,self.k2,self.c_rh,self.c_rh_min,self.ls0,self.c1,self.cs,self.ks,self.lambda0,self.km,self.kt,self.A,self.C,self.F0]
        #x = [self.k1,self.k2,self.c_rh,self.c_rh_min,self.lambda0,self.km,self.kt,self.A,self.C,self.F0]
        #x = [self.k1,self.k2,self.c_rh,self.c_rh_min,self.ls0,self.c1,self.cs,self.ks,self.lambda0,self.A,self.B,self.C,self.D,self.F0]
        x = [self.km,self.kt, self.c1]
        return x

    def get_initial_length(self):
        return (self.kt * self.delta - self.F0) / (self.kt + self.km)
        #return (self.kt * self.delta - self.F0 - self.c2 * self.lambda0) / (self.kt + self.km)

    def get_X0(self):
        lm0 = self.get_initial_length()
        X0 = [lm0,0,self.lambda0,self.ls0]
        return X0
