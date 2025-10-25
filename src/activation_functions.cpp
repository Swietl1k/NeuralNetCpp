#include "activation_functions.hpp"
#include "types.hpp"
#include <cmath>


DynamicMatrix relu(const DynamicMatrix& z) {
    return z.cwiseMax(0.0);
}

DynamicMatrix relu_derivative(const DynamicMatrix& z) {
    return (z.array() > 0.0).cast<RealType>();
}


DynamicMatrix sigmoid(const DynamicMatrix& z) {
    // numerically stable sigmoid computation
    return (z.array() >= 0).select(
        1.0 / (1.0 + (-z.array()).exp()),         // for z >= 0
        z.array().exp() / (1.0 + z.array().exp()) // for z < 0
    );
}

DynamicMatrix sigmoid_derivative(const DynamicMatrix& z) {
    return sigmoid(z).array() * (1.0 - sigmoid(z).array());
}


DynamicMatrix tanh(const DynamicMatrix& z) {
    return z.array().tanh();
}

DynamicMatrix tanh_derivative(const DynamicMatrix& z) {
    return 1.0 - z.array().tanh().square();
}


DynamicMatrix softmax(const DynamicMatrix& z) {
    DynamicMatrix shifted = z;
    DynamicVector row_max = z.rowwise().maxCoeff();
    shifted = (shifted.colwise() - row_max); // subtract max per row to avoid overflow

    DynamicMatrix exp_z = shifted.array().exp();
    DynamicVector sum_exp = exp_z.rowwise().sum();
    return exp_z.array().colwise() / sum_exp.array();
}