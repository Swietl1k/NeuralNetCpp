#pragma once

#include <Eigen/Dense>

using RealType = double;

using DynamicMatrix = Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>;
using DynamicVector = Eigen::Matrix<RealType, Eigen::Dynamic, 1>;
