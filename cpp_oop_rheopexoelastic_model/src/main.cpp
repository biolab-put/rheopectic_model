#include <iostream>
#include <cmath>
#include <vector>
#include "rheopecticmuscle.h"
#include <fstream>
#include <sstream>
#include <chrono>
std::vector<double> parabolic_twitches(const std::vector<double>& stim_dig, double twitch_duration, double twitch_delay, double twitch_amplitude, double sim_dt) {
    std::vector<double> single_twitch_t;
    for (double t = 0; t <= twitch_duration; t += sim_dt)
        single_twitch_t.push_back(t);

    std::vector<double> single_pulse;
    for (double t : single_twitch_t)
        single_pulse.push_back(twitch_amplitude * sin(M_PI * t / twitch_duration));

    std::vector<double> delay(static_cast<int>(twitch_delay / sim_dt), 0.0);
    std::vector<double> single_twitch(delay);
    single_twitch.insert(single_twitch.end(), single_pulse.begin(), single_pulse.end());

    std::vector<double> twitch_train(stim_dig.size(), 0.0);
    for (int i = 0; i < stim_dig.size(); i++) {
        if (stim_dig[i] == 1) {
            for (int j = 0; j < single_twitch.size(); j++)
                twitch_train[i + j] += single_twitch[j];
        }
    }

    return twitch_train;
}

std::vector<double> readFromCSV(const std::string& filename, const std::string& column_name) {
    std::vector<double> stim_dig;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return stim_dig;
    }

    std::string line;
    bool header = true;
    size_t selected_column_index = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string value;

        if (header) {
            // Read the header line to determine the index of the "stimDig" column
            std::string column;
            std::istringstream header_iss(line);
            while (std::getline(header_iss, column, ',')) {
                if (column == column_name) {
                    break;
                }
                selected_column_index++;
            }

            header = false;
        } else {
            // Read the "stimDig" value from the corresponding column
            size_t current_index = 0;
            while (std::getline(iss, value, ',')) {
                if (current_index == selected_column_index) {
                    try {
                        double dig = std::stod(value);
                        stim_dig.push_back(dig);
                    } catch (const std::exception& e) {
                        std::cerr << "Failed to parse value: " << value << std::endl;
                    }
                    break;
                }
                current_index++;
            }
        }
    }

    file.close();
    return stim_dig;
}

void save_vector_to_csv(const std::vector<double>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i];
        if (i != data.size() - 1) {
            file << ",";
        }
    }

    file.close();
    std::cout << "Vector data saved to file: " << filename << std::endl;
}


int main() {
    // Example usage
    double km = 9.27470775e+01;
    double kt = 2.69257705e+03;
    double m = 7.47363965e-03;
    double c_rh_min = 4.18334496e+01;
    double tr = 2.26529378e+01;
    double K = 2.19319711e+01;
    double Beta = -2.64438329e+00;
    double k1 = 1.32691853e+02;
    double k2 = 7.25133814e+01;
    double lambda0 = 2.91062475e-01;
    double A = 3.85402111e+00;
    double B = 3.24391366e-01;
    double G0 = 8.31611006e+00;
    double delta = 0.005;
    double sim_dt = 0.0001;
    double twitch_duration = 1.87171287e-02;
    double twitch_delay = 2.47903488e-03;
    double twitch_amplitude = 7.18294449e-02;
    std::string filename = "947kLHLf_trial_17";


    std::vector<double> stim_dig = readFromCSV(filename,"stimDig");
    std::vector<double> active_force = parabolic_twitches(stim_dig, twitch_duration, twitch_delay, twitch_amplitude, sim_dt);
    std::vector<double> time_vector(active_force.size());
    double time = 0.0;
    for (size_t i = 0; i < time_vector.size(); ++i) {
        time_vector[i] = time;
        time += sim_dt;
    }
    RheopecticMuscle muscle(km, kt, m, c_rh_min, tr, K, Beta, k1, k2, lambda0, A, B, G0, delta, sim_dt);
    std::vector<double> X0 = muscle.get_X0();
    auto start = std::chrono::high_resolution_clock::now();
    auto result = muscle.muscle_response(X0, time_vector, active_force);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << std::endl;
    save_vector_to_csv(result.first,"estimated_force.csv");
    std::cout << "Estimated Force size: "<< result.first.size() << std::endl;
    return 0;
}
