#include "initializers.hpp"
#include "types.hpp"
#include <random>
#include <cmath>
#include <Eigen/Dense>

DynamicMatrix xavier_init(int out, int in) {
    RealType scale = std::sqrt(6.0 / (in + out));
    return DynamicMatrix::Random(out, in) * scale;
}

DynamicMatrix he_init(int out, int in) {
    RealType scale = std::sqrt(RealType(2.0) / in);
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<RealType> normal_dist(0, scale);
    return DynamicMatrix::NullaryExpr(out, in, [&]() { return normal_dist(gen); });
}
