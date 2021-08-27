function dXdt = diff_solver(t,X,active_force,m,km,kt,c,sim_dt)
    lm=X(1); %x1
    dlm=X(2); %x2
    F_ce = active_force(1+round(t/sim_dt));
    x = [F_ce; lm; dlm];
    dXdt = [dlm ; MTU(x, m,km,kt,c)];
end