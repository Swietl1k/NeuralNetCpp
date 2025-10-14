#pragma once

#include <Eigen/Dense>
#include "hyper_params.hpp"

double cross_entropy_loss(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred, const std::vector<Eigen::MatrixXd>& weights, Regularization reg = Regularization::None, double lambda = 0.0);
Eigen::MatrixXd cross_entropy_gradient(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred);