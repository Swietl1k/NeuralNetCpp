#pragma once

#include "types.hpp"
#include <Eigen/Dense>

DynamicMatrix xavier_init(int out, int in);
DynamicMatrix he_init(int out, int in);