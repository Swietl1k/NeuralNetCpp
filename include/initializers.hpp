#pragma once

#include "types.hpp"
#include <Eigen/Dense>
#include <random> 

DynamicMatrix xavier_init(int out, int in, std::mt19937& gen);
DynamicMatrix he_init(int out, int in, std::mt19937& gen);