#pragma once

#include "types.hpp"
#include <Eigen/Dense>

DynamicMatrix relu(const DynamicMatrix& z);
DynamicMatrix relu_derivative(const DynamicMatrix& z);
DynamicMatrix sigmoid(const DynamicMatrix& z);
DynamicMatrix sigmoid_derivative(const DynamicMatrix& z);
DynamicMatrix tanh(const DynamicMatrix& z);
DynamicMatrix tanh_derivative(const DynamicMatrix& z);
DynamicMatrix softmax(const DynamicMatrix& z);  
DynamicMatrix softmax_derivative(const DynamicMatrix& z);