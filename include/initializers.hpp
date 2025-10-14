#pragma once

#include <Eigen/Dense>

Eigen::MatrixXd xavier_init(int out, int in);
Eigen::MatrixXd he_init(int out, int in);