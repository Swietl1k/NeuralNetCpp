#include "loss_functions.hpp"
#include "types.hpp"
#include <Eigen/Dense>

RealType cross_entropy_loss(
    const DynamicMatrix& y_true,
    const DynamicMatrix& y_pred,
    const std::vector<DynamicMatrix>& weights,
    Regularization reg,
    RealType lambda) {

    // lower bound epsilon clipping to avoid log(0)
    DynamicMatrix y_clipped = y_pred.array().max(1e-12);
    RealType loss = -(y_true.array() * y_clipped.array().log()).rowwise().sum().mean();

    RealType reg_term = 0.0;
    if (reg == Regularization::L2) {
        for (const auto& W : weights) {
            reg_term += (W.array().square().sum());
        }
        reg_term *= (lambda / (2.0 * y_true.rows()));
    } else if (reg == Regularization::L1) {
        for (const auto& W : weights) {
            reg_term += W.array().abs().sum();
        }
        reg_term *= (lambda / y_true.rows());
    }

    return loss + reg_term;
}

DynamicMatrix cross_entropy_gradient(const DynamicMatrix& y_true, const DynamicMatrix& y_pred) {
    return (y_pred - y_true) / y_true.rows(); // simplified gradient for softmax + cross-entropy
}
