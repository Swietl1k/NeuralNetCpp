#pragma once

#include <Eigen/Dense>

Eigen::MatrixXd relu(const Eigen::MatrixXd& z);
Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd& z);
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z);
Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd& z);
Eigen::MatrixXd tanh(const Eigen::MatrixXd& z);
Eigen::MatrixXd tanh_derivative(const Eigen::MatrixXd& z);
Eigen::MatrixXd softmax(const Eigen::MatrixXd& z);  
Eigen::MatrixXd softmax_derivative(const Eigen::MatrixXd& z);