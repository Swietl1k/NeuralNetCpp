#pragma once

#include "types.hpp"
#include "hyper_params.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>


class Network {
public:
    Network(int input_size, int output_size, const HyperParams& params, unsigned int seed = std::random_device{}());
    void train(const DynamicMatrix& X, const DynamicMatrix& X_val, const DynamicMatrix& y, const DynamicMatrix& y_val, bool save_accuracies = false, const std::string& output_filename = "accuracy_log.csv");
    std::tuple<RealType, RealType> evaluate(const DynamicMatrix& X, const DynamicMatrix& y); // returns loss and accuracy

private:
    HyperParams params;
    std::vector<int> layer_sizes;
    std::vector<DynamicMatrix> weights; // W[i]: weights between layer i and i+1
    std::vector<DynamicVector> biases;
    std::vector<DynamicMatrix> activations; 
    std::vector<DynamicMatrix> z_values; // pre-activation values
    std::vector<DynamicMatrix> dropout_masks;
    std::mt19937 gen;

    DynamicMatrix forward(const DynamicMatrix& X, bool training); // returns output layer activations
    void backward(const DynamicMatrix& X, const DynamicMatrix& y);    
    void build_layer_sizes(int input_size, int output_size);
    void initialize_weights();
    RealType compute_accuracy(const DynamicMatrix& y_true, const DynamicMatrix& y_pred);
};