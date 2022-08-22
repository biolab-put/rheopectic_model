function X = muscle_response(X0,time_vector, active_force,m,km,kt,c,sim_dt)
    X=ode4(@(t,X) diff_solver(t,X,active_force, m,km,kt,c,sim_dt),time_vector,X0);
end