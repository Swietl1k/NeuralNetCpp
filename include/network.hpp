#pragma once

#include "hyper_params.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>


class Network {
public:
    Network(int input_size, int output_size, const HyperParams& params, unsigned int seed = std::random_device{}());
    void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& X_val, const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_val, bool save_accuracies = false);
    std::tuple<double, double> evaluate(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y); // returns loss and accuracy

private:
    HyperParams params;
    std::vector<int> layer_sizes;
    std::vector<Eigen::MatrixXd> weights; // W[i]: weights between layer i and i+1
    std::vector<Eigen::VectorXd> biases;
    std::vector<Eigen::MatrixXd> activations; 
    std::vector<Eigen::MatrixXd> z_values; // pre-activation values
    std::vector<Eigen::MatrixXd> dropout_masks;
    std::mt19937 gen{std::random_device{}()};

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X, bool training); // returns output layer activations
    void backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y);    
    void build_layer_sizes(int input_size, int output_size);
    void initialize_weights();
    double compute_accuracy(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred);
};