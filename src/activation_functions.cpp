#include "activation_functions.hpp"
#include <cmath>


Eigen::MatrixXd relu(const Eigen::MatrixXd& z) {
    return z.cwiseMax(0.0);
}

Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd& z) {
    return (z.array() > 0.0).cast<double>();
}


Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z) {
    // numerically stable sigmoid computation
    return (z.array() >= 0).select(
        1.0 / (1.0 + (-z.array()).exp()),         // for z >= 0
        z.array().exp() / (1.0 + z.array().exp()) // for z < 0
    );
}

Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd& z) {
    return sigmoid(z).array() * (1.0 - sigmoid(z).array());
}


Eigen::MatrixXd tanh(const Eigen::MatrixXd& z) {
    return z.array().tanh();
}

Eigen::MatrixXd tanh_derivative(const Eigen::MatrixXd& z) {
    return 1.0 - z.array().tanh().square();
}


Eigen::MatrixXd softmax(const Eigen::MatrixXd& z) {
    Eigen::MatrixXd shifted = z;
    Eigen::VectorXd row_max = z.rowwise().maxCoeff();
    shifted = (shifted.colwise() - row_max); // subtract max per row to avoid overflow

    Eigen::MatrixXd exp_z = shifted.array().exp();
    Eigen::VectorXd sum_exp = exp_z.rowwise().sum();
    return exp_z.array().colwise() / sum_exp.array();
}