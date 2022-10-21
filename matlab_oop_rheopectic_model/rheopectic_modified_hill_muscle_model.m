classdef rheopectic_modified_hill_muscle_model
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        km
        kt
        m
        c
        cs
        ks
        ls0
        c1
        k1
        k2
        A
        B
        C
        D
        lambda0
        c_rh
        c_rh_min
        F0
        delta
        sim_dt
    end
    
    methods
        function self = rheopectic_modified_hill_muscle_model(km,kt,m,cs,ks,ls0,c_rh,c_rh_min,c1,k1,k2,A,B,C,D,lambda0,F0,delta,sim_dt)
            self.km = km;
            self.kt = kt;
            self.m = m;
            self.cs = cs;
            self.ks = ks;
            self.ls0 = ls0;
            self.c1 = c1;
            self.k1 = k1;
            self.k2 = k2;
            self.A = A;
            self.B = B;
            self.C = C;
            self.D = D;
            self.lambda0 = lambda0;
            self.c_rh = c_rh;
            self.c_rh_min = c_rh_min;
            self.F0 = F0;
            self.delta = delta;
            self.sim_dt = sim_dt;
        end
        
        function dXdt = solve_muscle_dynamics(self,t,X,active_force)
            lm = X(1);
            dlm_dt = X(2);
            Lambda = X(3);
            ls = X(4);
            dlm_dt_temp = max(0,dlm_dt);
            Lambda = max(min(Lambda,1),0);
            dLambda_dt = -self.k1 * (dlm_dt_temp.^self.A) * (Lambda.^self.B) + self.k2 *(dlm_dt_temp.^self.C) * (1-Lambda).^self.D;
            dls_dt = self.get_dls_dt(ls,active_force(1 + int32(t/self.sim_dt)));
            d2lm_dt = 1/self.m*(-(self.c_rh * Lambda * dlm_dt + self.c_rh_min * dlm_dt + self.c1 * dls_dt) - self.km*(sign(lm) * (lm.^2))+self.kt*(sign(self.delta-lm)* ((self.delta-lm).^2))-active_force(1 + int32(t/self.sim_dt)) - self.F0);
            dXdt = [dlm_dt;d2lm_dt;dLambda_dt;dls_dt];
        end

        function X = muscle_response(self,X0,time_vector,active_force)
           X=ode4(@(t,X) self.solve_muscle_dynamics(t,X,active_force),time_vector,X0);

           estimated_force = self.kt * sign(self.delta - X(:,1)) .* (abs(self.delta - X(:,1)) .^2) * 0.1;
           estimated_force = estimated_force(:) - estimated_force(1);
           estimated_force = max(estimated_force,0);
           X = [X,estimated_force];
        end
        
        function self = set_parameters(self,x)
            self.k1 = x(1);
            self.k2 = x(2);
            self.c_rh = x(3);
            self.c_rh_min = x(4);
            self.ls0 = x(5);
            self.c1 = x(6);
            self.cs = x(7);
            self.ks = x(8);
            self.lambda0 = x(9);
            self.A = x(10);
            self.B = x(11);
            self.C = x(12);
            self.D = x(13);
            self.km = x(14);
            self.kt = x(15);
            self.F0 = x(16);
        end
        
        function dls_dt = get_dls_dt(self,current_ls, current_active_force)
            dls_dt = 1/self.cs * (-(self.ks * current_ls) + current_active_force);
        end
        
        function [ x ] = get_parameters(self)
             x = [self.k1,self.k2,self.c_rh,self.c_rh_min,self.ls0,self.c1,self.cs,self.ks,self.lambda0,self.A,self.B,self.C,self.D,self.km,self.kt,self.F0];
        end
        
        function lm0 = get_initial_length(self)
            dls0 = self.get_dls_dt(self.ls0,0);
            a = self.kt - self.km;
            b = 2 * self.kt * self.delta;
            cc = self.kt*(self.delta.^2) - self.c1 * dls0 - self.F0;
            quadratic_discriminant = b.^2 - 4 * a * cc;
            lm01 = (-b + quadratic_discriminant.^(1/2)) / (2 * a);
            lm02 = (-b - quadratic_discriminant.^(1/2)) / (2 * a);
            if (lm01 > 0 && lm02 > 0) || (lm01 < 0 && lm02 < 0)
               lm0 = -1; 
            end
            if lm01 > 0
                lm0 = lm01;
                return;
            end
            if lm02 > 0
                lm0 = lm02;
                return;
            end
        end
        
        function [ X0 ] = get_X0(self)
            lm0 = self.get_initial_length();
            X0 = [lm0,0,self.lambda0,self.ls0];
        end
    end
end

