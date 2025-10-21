// STATIC NETWORK

#include "network.hpp"
#include "initializers.hpp"
#include "activation_functions.hpp"
#include "loss_functions.hpp"
#include "params.hpp"

#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>


StaticNetwork::StaticNetwork(unsigned int seed) 
    : gen(seed) {
    initialize_weights();
}

void StaticNetwork::train(
    const DynamicMatrix& X,
    const DynamicMatrix& y,
    const DynamicMatrix& X_val,
    const DynamicMatrix& y_val,
    bool save_accuracies) {

    std::vector<RealType> train_accuracies;
    std::vector<RealType> val_accuracies;

    int n_samples = TrainSamples - ValidationSamples;
    std::vector<int> indices(n_samples);
    for (int i = 0; i < n_samples; ++i) indices[i] = i;

    for (int epoch = 0; epoch < Epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), gen);
        RealType epoch_loss = 0.0;
        RealType epoch_accuracy = 0.0;
        int num_batches = 0;    

        for (int start = 0; start < n_samples; start += BatchSize) {
            if (start + BatchSize > n_samples) {
                // Skip incomplete batch for simplicity
                continue;
            }

            MatrixT<BatchSize, InputSize> X_batch;
            MatrixT<BatchSize, OutputSize> y_batch;
            for (int i = 0; i < BatchSize; ++i) {
                X_batch.row(i) = X.row(indices[start + i]);
                y_batch.row(i) = y.row(indices[start + i]);
            }

            MatrixT<BatchSize, OutputSize> y_pred = forward(X_batch, true);
            backward(y_batch);

            epoch_loss += cross_entropy_loss<BatchSize, OutputSize>(
                y_batch, y_pred, weights, Reg, Lambda
            ) * BatchSize;

            int correct = 0;
            for (int i = 0; i < BatchSize; ++i) {
                Eigen::Index pred_label, true_label;
                y_batch.row(i).maxCoeff(&true_label);
                y_pred.row(i).maxCoeff(&pred_label);
                if (pred_label == true_label) correct++;
            }
            epoch_accuracy += correct;
            num_batches++;
        }

        RealType train_loss = epoch_loss / (num_batches * BatchSize);
        RealType train_acc = epoch_accuracy / (num_batches * BatchSize);

        DynamicMatrix y_val_pred = forward_dynamic(X_val);
        RealType val_loss = cross_entropy_loss_dynamic(
            y_val, y_val_pred, weights, Reg, Lambda
        );
        RealType val_acc = compute_accuracy_dynamic(y_val, y_val_pred);

        if (save_accuracies) {
            train_accuracies.push_back(train_acc);
            val_accuracies.push_back(val_acc);
        }

        std::cout << "Epoch " << epoch+1
                << " | Train Loss: " << train_loss
                << " | Val Loss: " << val_loss
                << " | Train Acc: " << train_acc*100 << "%"
                << " | Val Acc: " << val_acc*100 << "%"
                << std::endl;
    }

    if (save_accuracies) {
        std::ofstream acc_file("../accuracies/accuracy_log.csv");
        acc_file << "epoch,train_acc,val_acc\n";
        for (size_t i = 0; i < train_accuracies.size(); ++i) {
            acc_file << (i + 1) << "," << train_accuracies[i] << "," << val_accuracies[i] << "\n";
        }
        acc_file.close();
    }
}

std::tuple<RealType, RealType> StaticNetwork::evaluate(
    const DynamicMatrix& X,
    const DynamicMatrix& y) {
    
    DynamicMatrix y_pred = forward_dynamic(X);
    RealType loss = cross_entropy_loss_dynamic(
        y, y_pred, weights, Reg, Lambda
    );
    RealType accuracy = compute_accuracy_dynamic(y, y_pred);
    
    return {loss, accuracy};
}


MatrixT<BatchSize, OutputSize> StaticNetwork::forward(
    const MatrixT<BatchSize, InputSize>& X, bool training) {

    std::get<0>(activations) = X;

    // hidden layer 1
    {
        auto& W = std::get<0>(weights);
        auto& b = std::get<0>(biases);
        auto z = ((X * W.transpose()).rowwise() + b.transpose()).eval();
        std::get<0>(z_values) = z;

        auto a = apply_activation(z, Act);
        auto [a_dropped, mask] = apply_dropout(a, training, DropoutRate, gen);
        std::get<0>(dropout_masks) = mask;
        std::get<1>(activations) = a_dropped;
    }

    // hidden layer 2
    {
        auto& W = std::get<1>(weights);
        auto& b = std::get<1>(biases);
        auto& a_prev = std::get<1>(activations);
        auto z = ((a_prev * W.transpose()).rowwise() + b.transpose()).eval();
        std::get<1>(z_values) = z;

        auto a = apply_activation(z, Act);
        auto [a_dropped, mask] = apply_dropout(a, training, DropoutRate, gen);
        std::get<1>(dropout_masks) = mask;
        std::get<2>(activations) = a_dropped;
    }

    // hidden layer 3
    {
        auto& W = std::get<2>(weights);
        auto& b = std::get<2>(biases);
        auto& a_prev = std::get<2>(activations);
        auto z = ((a_prev * W.transpose()).rowwise() + b.transpose()).eval();
        std::get<2>(z_values) = z;

        auto a = apply_activation(z, Act);
        auto [a_dropped, mask] = apply_dropout(a, training, DropoutRate, gen);
        std::get<2>(dropout_masks) = mask;
        std::get<3>(activations) = a_dropped;
    }

    // output layer
    {
        auto& W = std::get<3>(weights);
        auto& b = std::get<3>(biases);
        auto& a_prev = std::get<3>(activations);
        auto z = ((a_prev * W.transpose()).rowwise() + b.transpose()).eval();
        std::get<3>(z_values) = z;

        auto a = softmax(z);
        std::get<3>(dropout_masks) = MatrixT<BatchSize, OutputSize>::Ones();
        std::get<4>(activations) = a;
        
        return a;
    }
}


DynamicMatrix StaticNetwork::forward_dynamic(const DynamicMatrix& X) {
    auto& W0 = std::get<0>(weights);
    auto& b0 = std::get<0>(biases);
    DynamicMatrix z1 = (X * W0.transpose()).rowwise() + b0.transpose();
    DynamicMatrix a1 = apply_activation_dynamic(z1, Act);
    
    auto& W1 = std::get<1>(weights);
    auto& b1 = std::get<1>(biases);
    DynamicMatrix z2 = (a1 * W1.transpose()).rowwise() + b1.transpose();
    DynamicMatrix a2 = apply_activation_dynamic(z2, Act);
    
    auto& W2 = std::get<2>(weights);
    auto& b2 = std::get<2>(biases);
    DynamicMatrix z3 = (a2 * W2.transpose()).rowwise() + b2.transpose();
    DynamicMatrix a3 = apply_activation_dynamic(z3, Act);
    
    auto& W3 = std::get<3>(weights);
    auto& b3 = std::get<3>(biases);
    DynamicMatrix z_out = (a3 * W3.transpose()).rowwise() + b3.transpose();
    DynamicMatrix a_out = softmax_dynamic(z_out);
    
    return a_out;
}

void StaticNetwork::backward(const MatrixT<BatchSize, OutputSize>& y) {
    
    MatrixT<BatchSize, OutputSize> delta_out = std::get<4>(activations) - y;
    
    // layer 3 -> output
    auto& W3 = std::get<3>(weights);
    auto& a3 = std::get<3>(activations);
    MatrixT<OutputSize, L3> dW3 = (delta_out.transpose() * a3) / BatchSize;
    VectorT<OutputSize> db3 = delta_out.colwise().mean();
    if (Reg == Regularization::L2) {
        dW3 += (Lambda / BatchSize) * W3;
    }
    
    // backprop to layer 3
    MatrixT<BatchSize, L3> delta3 = delta_out * W3;
    delta3.array() *= std::get<2>(dropout_masks).array();
    delta3.array() *= apply_activation_derivative(std::get<2>(z_values), Act).array();
    
    // layer 2 -> layer 3
    auto& W2 = std::get<2>(weights);
    auto& a2 = std::get<2>(activations);
    MatrixT<L3, L2> dW2 = (delta3.transpose() * a2) / BatchSize;
    VectorT<L3> db2 = delta3.colwise().mean();
    if (Reg == Regularization::L2) {
        dW2 += (Lambda / BatchSize) * W2;
    }
    
    // backprop to layer 2
    MatrixT<BatchSize, L2> delta2 = delta3 * W2;
    delta2.array() *= std::get<1>(dropout_masks).array();
    delta2.array() *= apply_activation_derivative(std::get<1>(z_values), Act).array();
    
    // layer 1 -> layer 2
    auto& W1 = std::get<1>(weights);
    auto& a1 = std::get<1>(activations);
    MatrixT<L2, L1> dW1 = (delta2.transpose() * a1) / BatchSize;
    VectorT<L2> db1 = delta2.colwise().mean();
    if (Reg == Regularization::L2) {
        dW1 += (Lambda / BatchSize) * W1;
    }
    
    // backprop to layer 1
    MatrixT<BatchSize, L1> delta1 = delta2 * W1;
    delta1.array() *= std::get<0>(dropout_masks).array();
    delta1.array() *= apply_activation_derivative(std::get<0>(z_values), Act).array();
    
    // input -> layer 1
    auto& W0 = std::get<0>(weights);
    auto& a0 = std::get<0>(activations);
    MatrixT<L1, InputSize> dW0 = (delta1.transpose() * a0) / BatchSize;
    VectorT<L1> db0 = delta1.colwise().mean();
    if (Reg == Regularization::L2) {
        dW0 += (Lambda / BatchSize) * W0;
    }
    
    std::get<3>(weights) -= LearningRate * dW3;
    std::get<3>(biases) -= LearningRate * db3;
    
    std::get<2>(weights) -= LearningRate * dW2;
    std::get<2>(biases) -= LearningRate * db2;
    
    std::get<1>(weights) -= LearningRate * dW1;
    std::get<1>(biases) -= LearningRate * db1;
    
    std::get<0>(weights) -= LearningRate * dW0;
    std::get<0>(biases) -= LearningRate * db0;
}

void StaticNetwork::initialize_weights() {
    if (Init == InitType::Xavier) {
        std::get<0>(weights) = xavier_init<L1, InputSize>(gen);
        std::get<1>(weights) = xavier_init<L2, L1>(gen);
        std::get<2>(weights) = xavier_init<L3, L2>(gen);
        std::get<3>(weights) = xavier_init<OutputSize, L3>(gen);
    } else {
        std::get<0>(weights) = he_init<L1, InputSize>(gen);
        std::get<1>(weights) = he_init<L2, L1>(gen);
        std::get<2>(weights) = he_init<L3, L2>(gen);
        std::get<3>(weights) = he_init<OutputSize, L3>(gen);
    }
    
    std::get<0>(biases) = VectorT<L1>::Zero();
    std::get<1>(biases) = VectorT<L2>::Zero();
    std::get<2>(biases) = VectorT<L3>::Zero();
    std::get<3>(biases) = VectorT<OutputSize>::Zero();
}

RealType StaticNetwork::compute_accuracy_dynamic(
    const DynamicMatrix& y_true,
    const DynamicMatrix& y_pred) {
    
    int correct = 0;
    int n_samples = y_true.rows();
    
    for (int i = 0; i < n_samples; i++) {
        Eigen::Index pred_label, true_label;
        y_true.row(i).maxCoeff(&true_label);
        y_pred.row(i).maxCoeff(&pred_label);
        if (pred_label == true_label) correct++;
    }
    
    return static_cast<RealType>(correct) / n_samples;
}