//STATIC NETWORK

#pragma once
#include <Eigen/Dense>
#include <tuple>

#include "params.hpp"

template<int Rows, int Cols, typename... Weights>
constexpr RealType cross_entropy_loss(
    const MatrixT<Rows, Cols>& y_true,
    const MatrixT<Rows, Cols>& y_pred,
    const std::tuple<Weights...>& weights,
    Regularization reg = Regularization::None,
    RealType lambda = 0.0) {
    // clipping to avoid log(0)
    MatrixT<Rows, Cols> y_clipped = y_pred.array().max(RealType(1e-12));

    RealType loss = -(y_true.array() * y_clipped.array().log()).rowwise().sum().mean();

    RealType reg_term = 0.0;
    if (reg == Regularization::L2) {
        std::apply([&](const auto&... W) {
            ((reg_term += (W.array().square().sum())), ...);
        }, weights);
        reg_term *= (lambda / (2.0 * static_cast<RealType>(Rows)));
    } else if (reg == Regularization::L1) {
        std::apply([&](const auto&... W) {
            ((reg_term += (W.array().abs().sum())), ...);
        }, weights);
        reg_term *= (lambda / static_cast<RealType>(Rows));
    }

    return loss + reg_term;
}

MatrixT<BatchSize, OutputSize> cross_entropy_gradient(const MatrixT<BatchSize, OutputSize>& y_true, const MatrixT<BatchSize, OutputSize>& y_pred);


template<typename WeightsTuple>
RealType cross_entropy_loss_dynamic(
    const DynamicMatrix& y_true,
    const DynamicMatrix& y_pred,
    const WeightsTuple& weights,
    Regularization reg,
    RealType lambda) {
    
    const RealType epsilon = 1e-12;
    DynamicMatrix y_clipped = y_pred.array().max(epsilon).matrix();
    
    RealType loss = -(y_true.array() * y_clipped.array().log()).rowwise().sum().mean();
    
    if (lambda > 0.0) {
        RealType reg_term = 0.0;
        
        if (reg == Regularization::L2) {
            std::apply([&](const auto&... W) {
                ((reg_term += W.array().square().sum()), ...);
            }, weights);
            loss += (lambda / (2.0 * y_true.rows())) * reg_term;
        } else if (reg == Regularization::L1) {
            std::apply([&](const auto&... W) {
                ((reg_term += W.array().abs().sum()), ...);
            }, weights);
            loss += (lambda / y_true.rows()) * reg_term;
        }
    }
    
    return loss;
}