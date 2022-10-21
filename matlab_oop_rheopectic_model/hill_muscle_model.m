classdef hill_muscle_model
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        km
        kt
        m
        c
        delta
        sim_dt
    end
    
    methods
        function self = hill_muscle_model(km,kt,m,c,delta,sim_dt)
            self.km = km;
            self.kt = kt;
            self.m = m;
            self.c = c;
            self.delta = delta;
            self.sim_dt = sim_dt;
        end
        
        function dXdt = solve_muscle_dynamics(self,t,X,active_force)
            lm = X(1);
            dlm_dt = X(2);
            d2lm_dt = 1/self.m*(-self.c*dlm_dt-self.km*lm+self.kt*(self.delta-lm)-active_force(1 + int32(t/self.sim_dt)));
            dXdt = [dlm_dt ; d2lm_dt];
        end

        function X = muscle_response(self,X0,time_vector,active_force)
           X=ode4(@(t,X) self.solve_muscle_dynamics(t,X,active_force),time_vector,X0);
           estimated_force = self.delta - X(:,1);
           estimated_force = estimated_force(:) - estimated_force(1);
           X = [X,estimated_force];
        end
        
        function self = set_parameters(self,x)
            self.km = x(1);
            self.kt = x(2);
            self.m = x(3);
            self.c = x(4);
        end
        
        function [ x ] = get_parameters(self)
            x = [self.km,self.kt,self.m,self.c];
        end
        
        function lm0 = get_initial_length(self)
            lm0 = self.kt * self.delta / (self.kt + self.km);
        end
        
        function [ X0 ] = get_X0(self)
            lm0 = self.get_initial_length();
            X0 = [lm0,0];
        end
    end
end

