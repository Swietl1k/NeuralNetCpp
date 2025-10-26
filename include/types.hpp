#pragma once

#include <Eigen/Dense>

using RealType = float;

using DynamicMatrix = Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>;
using DynamicVector = Eigen::Matrix<RealType, Eigen::Dynamic, 1>;
