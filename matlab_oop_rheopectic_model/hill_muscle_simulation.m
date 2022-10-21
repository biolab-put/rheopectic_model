sim_time = 0.8;
sim_dt = 0.0001;
time_vector = (0:sim_dt:sim_time);
twitch_duration = 0.01;
twitch_delay = 0.0025;
twitch_frequency = 40;
twitch_amplitude = 1;

muscle_model = create_hill_model(sim_dt);
active_force = parabolic_twitch(time_vector,twitch_duration,twitch_delay, twitch_amplitude, twitch_frequency, sim_dt);
active_force(1,0.52/sim_dt:end) = 0;
X0 = muscle_model.get_X0();
muscle_output = muscle_model.muscle_response(X0,time_vector,active_force);
plot(muscle_output(:,3)); %plot force
