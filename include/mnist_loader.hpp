#pragma once 

#include <Eigen/Dense>
#include <string>
#include <tuple>

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> load_fashion_mnist_csv(const std::string& filename);