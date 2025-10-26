#include "initializers.hpp"
#include "types.hpp"
#include <random>
#include <cmath>
#include <Eigen/Dense>

DynamicMatrix xavier_init(int out, int in, std::mt19937& gen) {
    RealType scale = std::sqrt(6.0 / (in + out));
    std::uniform_real_distribution<RealType> uniform_dist(-scale, scale);
    return DynamicMatrix::NullaryExpr(out, in, [&]() { return uniform_dist(gen); });
}
DynamicMatrix he_init(int out, int in, std::mt19937& gen) {
    RealType scale = std::sqrt(RealType(2.0) / in);
    std::normal_distribution<RealType> normal_dist(0, scale);
    return DynamicMatrix::NullaryExpr(out, in, [&]() { return normal_dist(gen); });
}