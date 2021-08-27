function square_train = square_twitch(t,twitch_duration,twitch_delay,twitch_amplitude, twitch_frequency, sim_dt)
%square_train = twitch_amplitude*rectpuls(t - twitch_duration/2,twitch_duration);
    single_twitch_t = 0:sim_dt:twitch_duration;
    pulse = @() twitch_amplitude*rectpuls(single_twitch_t - twitch_duration/2,twitch_duration-sim_dt);
    single_twitch = pulse();
    train_twitch_t = twitch_delay:1/twitch_frequency:t(end);
    square_train = pulstran(t,train_twitch_t,single_twitch,1/sim_dt,'nearest');
end