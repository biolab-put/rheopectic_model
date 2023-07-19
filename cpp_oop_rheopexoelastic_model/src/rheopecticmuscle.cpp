#include "RheopecticMuscle.h"
#include <iostream>
#include <cmath>

RheopecticMuscle::RheopecticMuscle(double km, double kt, double m, double c_rh_min, double tr, double K, double Beta, double k1, double k2, double lambda0, double A, double B, double G0, double delta, double sim_dt)
    : km(km), kt(kt), m(m), c_rh_min(c_rh_min), tr(tr), K(K), Beta(Beta), k1(k1), k2(k2), lambda0(lambda0), A(A), B(B), G0(G0), delta(delta), sim_dt(sim_dt) {}


int RheopecticMuscle::sign(double value) {
    if (value > 0) {
        return 1;
    } else if (value < 0) {
        return -1;
    } else {
        return 0;
    }
}

std::vector<double> RheopecticMuscle::solve_muscle_dynamics(double t, const std::vector<double>& X, const std::vector<double>& active_force, const std::vector<double>& X0) {
    double lm = X[0];
    double dlm_dt = X[1];
    double Lambda = X[2];
    double rh = X[3];

    if (Lambda < 0) {
        Lambda = 0;
    } else if (Lambda > 1) {
        Lambda = 1;
    }
    if (lm < 0) {
        lm = 0;
    } else if (lm > X0[0]) {
        lm = X0[0];
    }
    if ((lm <= 0 && dlm_dt < 0) || (lm >= X0[0] && dlm_dt > 0)) {
        dlm_dt = 0;
    }
    double dGamma_dt = dlm_dt;
    double dLambda_dt = -k1 * sign(dGamma_dt) * std::pow(std::abs(dGamma_dt), A) * Lambda + k2 * sign(dGamma_dt) * std::pow(std::abs(dGamma_dt), B) * (1 - Lambda);

    double c_rh = c_rh_min * std::pow((1 + K * Lambda), Beta);
    if ((Lambda <= 0 && dLambda_dt < 0) || (Lambda >= 1 && dLambda_dt > 0)) {
        dLambda_dt = 0;
    }
    double rheopectic_force = rh;
    double d2lm_dt = 1 / m * (-rheopectic_force - km * sign(lm) * lm + kt * sign(delta - lm) * (delta - lm) - std::max(0.0, active_force[static_cast<int>(t / sim_dt)]));

    double drh_dt = G0 * (dlm_dt + d2lm_dt * tr - rh / c_rh);

    return { dlm_dt, d2lm_dt, dLambda_dt, drh_dt };
}


std::vector<double> RheopecticMuscle::elementwise_add(const std::vector<double>& v1, const std::vector<double>& v2) {
    //assert(v1.size() == v2.size());
    std::vector<double> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

// Element-wise scalar multiplication of a vector
std::vector<double> RheopecticMuscle::elementwise_scalar_multiply(double scalar, const std::vector<double>& v) {
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = scalar * v[i];
    }
    return result;
}

std::pair<std::vector<double>, std::vector<std::vector<double>>> RheopecticMuscle::muscle_response(const std::vector<double>& X0, const std::vector<double>& time_vector, const std::vector<double>& active_force) {
    std::vector<std::vector<double>> solution(time_vector.size(), std::vector<double>(X0.size(), 0.0));
    solution[0] = X0;

    for (size_t n = 1; n < time_vector.size(); ++n) {
        std::vector<double> X = solution[n - 1];
        double t = time_vector[n];
        double dt = sim_dt;

        // RK4 integration
        std::vector<double> k1 = solve_muscle_dynamics(t, X, active_force, X0);
        std::vector<double> k2 = solve_muscle_dynamics(t + dt / 2, elementwise_add(X, elementwise_scalar_multiply(dt / 2, k1)), active_force, X0);
        std::vector<double> k3 = solve_muscle_dynamics(t + dt / 2, elementwise_add(X, elementwise_scalar_multiply(dt / 2, k2)), active_force, X0);
        std::vector<double> k4 = solve_muscle_dynamics(t + dt, elementwise_add(X, elementwise_scalar_multiply(dt, k3)), active_force, X0);

        std::vector<double> new_X(X0.size(), 0.0);
        for (size_t i = 0; i < X0.size(); ++i) {
            new_X[i] = X[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        }
        solution[n] = new_X;
    }

    std::vector<double> estimated_force(time_vector.size(), 0.0);
    for (size_t i = 0; i < time_vector.size(); ++i) {
        estimated_force[i] = kt * sign(delta - solution[i][0]) * std::abs(delta - solution[i][0]);
    }

    double initial_force = estimated_force[0];
    for (double& force : estimated_force) {
        force -= initial_force;
    }

    return { estimated_force, solution };
}

/*
std::pair<std::vector<double>, std::vector<std::vector<double>>> RheopecticMuscle::muscle_response(const std::vector<double>& X0, const std::vector<double>& time_vector, const std::vector<double>& active_force) {
    std::vector<std::vector<double>> solution(time_vector.size(), std::vector<double>(X0.size(), 0.0));
    solution[0] = X0;

    for (size_t n = 1; n < time_vector.size(); ++n) {
        std::vector<double> X = solution[n - 1];
        double t = time_vector[n];

        std::vector<double> derivative = solve_muscle_dynamics(t, X, active_force, X0);
        std::vector<double> new_X(X0.size(), 0.0);
        for (size_t i = 0; i < X0.size(); ++i) {
            new_X[i] = X[i] + sim_dt * derivative[i];
        }
        solution[n] = new_X;
    }

    std::vector<double> estimated_force(time_vector.size(), 0.0);
    for (size_t i = 0; i < time_vector.size(); ++i) {
        estimated_force[i] = kt * std::signbit(delta - solution[i][0]) * std::abs(delta - solution[i][0]);
    }
    double initial_force = estimated_force[0];
    for (double& force : estimated_force) {
        force -= initial_force;
    }

    return { estimated_force, solution };
}
*/
void RheopecticMuscle::set_parameters(const std::vector<double>& x) {
    km = x[0];
    kt = x[1];
    m = x[2];
    c_rh_min = x[3];
    tr = x[4];
    K = x[5];
    Beta = x[6];
    k1 = x[7];
    k2 = x[8];
    A = x[9];
    B = x[10];
    G0 = x[11];
    lambda0 = x[12];
}

std::vector<double> RheopecticMuscle::get_parameters() {
    return { km, kt, m, c_rh_min, tr, K, Beta, k1, k2, A, B, G0, lambda0 };
}

double RheopecticMuscle::get_initial_length() {
    return kt * delta / (km + kt);
}

std::vector<double> RheopecticMuscle::get_X0() {
    double lm0 = get_initial_length();
    return { lm0, 0.0, lambda0, 0.0 };
}

double RheopecticMuscle::get_stiffness_ratio(const std::vector<double>& x) {
    double km = x[0];
    double kt = x[1];
    return kt / km;
}
