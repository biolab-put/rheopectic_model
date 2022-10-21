sim_time = 1.8;
sim_dt = 0.0001;
time_vector = (0:sim_dt:sim_time);
twitch_duration = 0.01;
twitch_delay = 0.0025;
twitch_frequency = 40;
low_frequency = 35;
high_frequency = 75;
twitch_amplitude = 1;
muscle_model = create_rheopectic_modified_hill_muscle_model(sim_dt);

high_freq_duration = int32((23 * ((1/high_frequency + sim_dt/1.5)))/sim_dt);
low_freq_duration = int32((19 * ((1/low_frequency - sim_dt/1.5)))/sim_dt);
input_1_low_freq = parabolic_twitch(time_vector(1:low_freq_duration),twitch_duration,twitch_delay,twitch_amplitude, low_frequency, sim_dt);
input_1_high_freq = parabolic_twitch(time_vector(low_freq_duration:low_freq_duration+high_freq_duration),twitch_duration,twitch_delay,twitch_amplitude, high_frequency, sim_dt);
input_1_low_freq_end = parabolic_twitch(time_vector(low_freq_duration+high_freq_duration:2*low_freq_duration+high_freq_duration),twitch_duration,twitch_delay,twitch_amplitude, low_frequency, sim_dt);

zeros_N = length(time_vector)-(2*low_freq_duration+high_freq_duration);
zeros_data = zeros(1,zeros_N);

active_force = [input_1_low_freq,input_1_high_freq,input_1_low_freq_end,zeros_data];
X0 = muscle_model.get_X0();
muscle_output = muscle_model.muscle_response(X0,time_vector,active_force);
plot(muscle_output(:,5));
