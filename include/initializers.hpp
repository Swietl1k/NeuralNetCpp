#pragma once
#include "params.hpp"
#include <random>
#include <cmath>

template<int Out, int In>
MatrixT<Out, In> xavier_init(std::mt19937& gen) {
    RealType scale = std::sqrt(RealType(6) / (In + Out));
    std::uniform_real_distribution<RealType> dist(-scale, scale);
    return MatrixT<Out, In>::NullaryExpr([&]() { return dist(gen); });
}

template<int Out, int In>
MatrixT<Out, In> he_init(std::mt19937& gen) {
    RealType scale = std::sqrt(RealType(2) / In);
    std::normal_distribution<RealType> dist(0, scale);
    return MatrixT<Out, In>::NullaryExpr([&]() { return dist(gen); });
}
