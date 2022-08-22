%% Parameters
% dwie zmienne stanu x_1 = lm i x_2 = dlm/dt
X0 = [0,0]; %stan pocz?tkowy, pocz?tkowe wyd?u?enie i pr?dko??
sim_dt = 0.001; %interwa? czasowy
sim_time = 0.9+0.3; %czas symulacji
time_vector = 0:sim_dt:sim_time-sim_dt;
twitch_duration = 0.025;
twitch_delay = 0.06;
twitch_frequency = 0;
twitch_amplitude = 10;
km = 0.1;
kt = 1;
c = 0.03;
m = 0.000650;

%% input signal
active_force = parabolic_twitch(time_vector,twitch_duration,twitch_delay, twitch_amplitude, twitch_frequency, sim_dt);
%active_force = square_twitch(time_vector,twitch_duration,twitch_delay, twitch_amplitude, twitch_frequency, sim_dt);
%% muscle response
X = muscle_response(X0,time_vector, active_force, m, km,kt,c, sim_dt);

%% plot lm
plot(time_vector,X(:,1));
xlabel('Time [s]');
ylabel('lm [m]'); %Rozci?gni?cie mi??nia
grid on;

%% plot force
%Przej?cie z rozci?gni?cia mi??nia na si??
output_force = X(:,1) .* kt;
output_force = output_force.*1000; %przeskalowanie do miliniutonów
plot(time_vector,output_force);
xlabel('Time [s]');
ylabel('Force [mN]'); % Si?a na wykresach z artyku?u jest w mininiutonach

%% plot input
plot(time_vector,active_force);

%% Load data for optimalization

trial_data = readtable('1006j_trial_0');
reference_muscle_force = trial_data.force;
stimulation = trial_data.stimDig;
zeros_time = sim_time - 0.5;
reference_muscle_force = reference_muscle_force(1:sim_time/sim_dt);
%reference_muscle_force = reference_muscle_force - min(reference_muscle_force);
reference_muscle_force = reference_muscle_force - reference_muscle_force(1);
reference_muscle_force(zeros_time/sim_dt:end) = zeros(1,round(((sim_time-zeros_time)/sim_dt)+1));


%% Optimalization

x0 = [km, kt, m, c];
%options = optimset('GradObj', 'off' ,'Display', 'iter', 'HessUpdate', 'bfgs', 'GoalsExactAchieve',1);
%x = fminlbfgs(@objectivefcn1, x0, options)
x = fminsearch(@(x) objectivefcn(x,X0, time_vector, active_force,reference_muscle_force, sim_dt),x0)

%% Plot result

km = x(1);
kt = x(2);
m = x(3);
c = x(4);
X = muscle_response(X0,time_vector, active_force, m, km,kt,c, sim_dt);

output_force = X(:,1) .* kt;
output_force = output_force.*1000; %przeskalowanie do miliniutonów
plot(time_vector,output_force);
hold on;
plot(time_vector,reference_muscle_force);
xlabel('Time [s]');
ylabel('Force [mN]'); % Si?a na wykresach z artyku?u jest w mininiutonach
grid on;


