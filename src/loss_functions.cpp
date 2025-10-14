#include "loss_functions.hpp"

#include <Eigen/Dense>

double cross_entropy_loss(
    const Eigen::MatrixXd& y_true,
    const Eigen::MatrixXd& y_pred,
    const std::vector<Eigen::MatrixXd>& weights,
    Regularization reg,
    double lambda) {

    // lower bound epsilon clipping to avoid log(0)
    Eigen::MatrixXd y_clipped = y_pred.array().max(1e-12);
    double loss = -(y_true.array() * y_clipped.array().log()).rowwise().sum().mean();

    double reg_term = 0.0;
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

Eigen::MatrixXd cross_entropy_gradient(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) {
    return (y_pred - y_true) / y_true.rows(); // simplified gradient for softmax + cross-entropy
}
