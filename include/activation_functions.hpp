#pragma once

#include "params.hpp"
#include <Eigen/Dense>
#include <random>


template<int Rows, int Cols>
MatrixT<Rows, Cols> relu(const MatrixT<Rows, Cols>& z) {
    return z.cwiseMax(0.0);
}

template<int Rows, int Cols>
MatrixT<Rows, Cols> relu_derivative(const MatrixT<Rows, Cols>& z) {
    return (z.array() > 0.0).template cast<RealType>();
}


template<int Rows, int Cols>
MatrixT<Rows, Cols> sigmoid(const MatrixT<Rows, Cols>& z) {
    // numerically stable sigmoid computation
    return (z.array() >= 0).select(
        1.0 / (1.0 + (-z.array()).exp()),         // for z >= 0
        z.array().exp() / (1.0 + z.array().exp()) // for z < 0
    );
}

template<int Rows, int Cols>
MatrixT<Rows, Cols> sigmoid_derivative(const MatrixT<Rows, Cols>& z) {
    MatrixT<Rows, Cols> sig = sigmoid(z);
    return sig.array() * (1.0 - sig.array());
}


template<int Rows, int Cols>
MatrixT<Rows, Cols> tanh(const MatrixT<Rows, Cols>& z) {
    return z.array().tanh();
}

template<int Rows, int Cols>
MatrixT<Rows, Cols> tanh_derivative(const MatrixT<Rows, Cols>& z) {
    return 1.0 - z.array().tanh().square();
}


template<int Rows, int Cols>
MatrixT<Rows, Cols> softmax(const MatrixT<Rows, Cols>& z) {
    MatrixT<Rows, Cols> shifted = z;
    VectorT<Rows> row_max = z.rowwise().maxCoeff();
    shifted = (shifted.colwise() - row_max); // subtract max per row to avoid overflow

    MatrixT<Rows, Cols> exp_z = shifted.array().exp();
    VectorT<Rows> sum_exp = exp_z.rowwise().sum();
    return exp_z.array().colwise() / sum_exp.array();
}

inline DynamicMatrix softmax_dynamic(const DynamicMatrix& z) {
    DynamicMatrix shifted = z;
    DynamicVector row_max = z.rowwise().maxCoeff();
    shifted = (shifted.colwise() - row_max);
    
    DynamicMatrix exp_z = shifted.array().exp().matrix();
    DynamicVector sum_exp = exp_z.rowwise().sum();
    return exp_z.array().colwise() / sum_exp.array();
}

template<int Rows, int Cols>
MatrixT<Rows, Cols> apply_activation(
    const MatrixT<Rows, Cols>& z,
    Activation activation) {
    
    switch (activation) {
        case Activation::ReLU:
            return relu(z);
        case Activation::Sigmoid:
            return sigmoid(z);
        case Activation::Tanh:
            return tanh(z);
        default:
            return z;
    }
}

template<int Rows, int Cols>
MatrixT<Rows, Cols> apply_activation_derivative(
    const MatrixT<Rows, Cols>& z,
    Activation activation) {
    
    switch (activation) {
        case Activation::ReLU:
            return relu_derivative(z);
        case Activation::Sigmoid:
            return sigmoid_derivative(z);
        case Activation::Tanh:
            return tanh_derivative(z);
        default:
            return MatrixT<Rows, Cols>::Ones();
    }
}

template<int Rows, int Cols>
std::pair<MatrixT<Rows, Cols>, MatrixT<Rows, Cols>> apply_dropout(
    const MatrixT<Rows, Cols>& activation,
    bool training,
    RealType dropout_rate,
    std::mt19937& gen) {
    
    if (training && dropout_rate > 0.0) {
        std::bernoulli_distribution dropout_dist(1.0 - dropout_rate);
        MatrixT<Rows, Cols> mask = MatrixT<Rows, Cols>::NullaryExpr(
            [&]() { return dropout_dist(gen) ? 1.0 : 0.0; });
        
        MatrixT<Rows, Cols> dropped = activation.array() * mask.array() / (1.0 - dropout_rate);
        return {dropped, mask};
    } else {
        return {activation, MatrixT<Rows, Cols>::Ones()};
    }
}


inline DynamicMatrix apply_activation_dynamic(
    const DynamicMatrix& z,
    Activation activation) {
    
    switch (activation) {
        case Activation::ReLU:
            return z.cwiseMax(0.0);
        case Activation::Sigmoid: {
            return (z.array() >= 0).select(
                1.0 / (1.0 + (-z.array()).exp()),
                z.array().exp() / (1.0 + z.array().exp())
            ).matrix();
        }
        case Activation::Tanh:
            return z.array().tanh().matrix();
        default:
            return z;
    }
}
