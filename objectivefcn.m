function sse = objectivefcn(x,time_vector,active_force, reference_muscle_force, sim_dt)

km = x(1);
kt = x(2);
m = x(3);
c = x(4);
X0 = [0,0];
X = muscle_response(X0,time_vector, active_force, m, km,kt,c, sim_dt);
estimated_muscle_force = X(:,1);

%change from lm to force
estimated_muscle_force = estimated_muscle_force .* kt;
%change from [mN] to [N]
estimated_muscle_force = estimated_muscle_force.*1000;

sse = sum((estimated_muscle_force - reference_muscle_force).^2);