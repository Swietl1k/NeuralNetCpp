#pragma once

#include "hyper_params.hpp"
#include "types.hpp"
#include <Eigen/Dense>

RealType cross_entropy_loss(const DynamicMatrix& y_true, const DynamicMatrix& y_pred, const std::vector<DynamicMatrix>& weights, Regularization reg = Regularization::None, RealType lambda = 0.0);
DynamicMatrix cross_entropy_gradient(const DynamicMatrix& y_true, const DynamicMatrix& y_pred);