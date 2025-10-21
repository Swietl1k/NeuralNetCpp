#pragma once

#include "params.hpp"

#include <Eigen/Dense>
#include <random>
#include <tuple>



class StaticNetwork {
public:
    explicit StaticNetwork(unsigned int seed = std::random_device{}());

    // train with dynamic-sized data
    void train(
        const DynamicMatrix& X,
        const DynamicMatrix& y,
        const DynamicMatrix& X_val,
        const DynamicMatrix& y_val,
        bool save_accuracies = false);

    // evaluate with dynamic-sized data
    std::tuple<RealType, RealType> evaluate(
        const DynamicMatrix& X,
        const DynamicMatrix& y);

private:
    std::mt19937 gen;

    // static-sized weights and biases (on the stack)
    std::tuple<
        MatrixT<L1, InputSize>,
        MatrixT<L2, L1>,
        MatrixT<L3, L2>,
        MatrixT<OutputSize, L3>
    > weights;

    std::tuple<
        VectorT<L1>,
        VectorT<L2>,
        VectorT<L3>,
        VectorT<OutputSize>
    > biases;

    // static-sized activations for batch processing
    std::tuple<
        MatrixT<BatchSize, InputSize>,
        MatrixT<BatchSize, L1>,
        MatrixT<BatchSize, L2>,
        MatrixT<BatchSize, L3>,
        MatrixT<BatchSize, OutputSize>
    > activations;

    std::tuple<
        MatrixT<BatchSize, L1>,
        MatrixT<BatchSize, L2>,
        MatrixT<BatchSize, L3>,
        MatrixT<BatchSize, OutputSize>
    > z_values;

    std::tuple<
        MatrixT<BatchSize, L1>,
        MatrixT<BatchSize, L2>,
        MatrixT<BatchSize, L3>,
        MatrixT<BatchSize, OutputSize>
    > dropout_masks;

    MatrixT<BatchSize, OutputSize> forward(
        const MatrixT<BatchSize, InputSize>& X, 
        bool training);
    void backward(const MatrixT<BatchSize, OutputSize>& y);
    void initialize_weights();
    
    DynamicMatrix forward_dynamic(const DynamicMatrix& X);
    RealType compute_accuracy_dynamic(
        const DynamicMatrix& y_true,
        const DynamicMatrix& y_pred);
};