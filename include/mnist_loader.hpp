#pragma once 

#include "types.hpp"
#include <Eigen/Dense>
#include <string>
#include <tuple>

std::tuple<DynamicMatrix, DynamicMatrix> load_fashion_mnist_csv(const std::string& filename);