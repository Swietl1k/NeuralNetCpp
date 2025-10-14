#include "initializers.hpp"
#include <random>
#include <cmath>

Eigen::MatrixXd xavier_init(int out, int in) {
    double scale = std::sqrt(6.0 / (in + out));
    return Eigen::MatrixXd::Random(out, in) * scale;
}

Eigen::MatrixXd he_init(int out, int in) {
    double scale = std::sqrt(2.0 / in);
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> normal_dist(0, scale);
    return Eigen::MatrixXd::NullaryExpr(out, in, [&]() { return normal_dist(gen); });
}
