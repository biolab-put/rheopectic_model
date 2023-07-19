#ifndef RHEOPECTICMUSCLE_H
#define RHEOPECTICMUSCLE_H

#include <vector>

class RheopecticMuscle {
private:
    double k1, k2, m, km, kt, lambda0, c_rh_min, tr, K, Beta, delta, sim_dt;
    double A, B, G0;
    std::vector<double> elementwise_add(const std::vector<double>& v1, const std::vector<double>& v2);
    std::vector<double> elementwise_scalar_multiply(double scalar, const std::vector<double>& v);
    int sign(double value);
public:
    RheopecticMuscle(double km, double kt, double m, double c_rh_min, double tr, double K, double Beta, double k1, double k2, double lambda0, double A, double B, double G0, double delta, double sim_dt);

    std::vector<double> solve_muscle_dynamics(double t, const std::vector<double>& X, const std::vector<double>& active_force, const std::vector<double>& X0);

    std::pair<std::vector<double>, std::vector<std::vector<double>>> muscle_response(const std::vector<double>& X0, const std::vector<double>& time_vector, const std::vector<double>& active_force);

    void set_parameters(const std::vector<double>& x);

    std::vector<double> get_parameters();

    double get_initial_length();

    std::vector<double> get_X0();

    static double get_stiffness_ratio(const std::vector<double>& x);
};

#endif
