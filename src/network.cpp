#include "network.hpp"
#include "initializers.hpp"
#include "activation_functions.hpp"
#include "loss_functions.hpp"

#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>

Network::Network(int input_size, int output_size, const HyperParams& params, unsigned int seed) : params(params), gen(seed) {
    build_layer_sizes(input_size, output_size);
    initialize_weights();
}

std::tuple<double, double> Network::evaluate(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) {
    Eigen::MatrixXd y_pred = forward(X, false);
    double loss = cross_entropy_loss(y, y_pred, weights, params.regularization, params.lambda);
    int correct = 0;

    for (int i = 0; i < X.rows(); i++) {
        Eigen::Index pred_label, true_label;
        y.row(i).maxCoeff(&true_label);
        y_pred.row(i).maxCoeff(&pred_label);
        if (pred_label == true_label) correct++;
    }

    double accuracy = static_cast<double>(correct) / X.rows();
    return {loss, accuracy};
}

void Network::train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& X_val, const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_val, bool save_accuracies) {
    std::vector<double> train_accuracies;
    std::vector<double> val_accuracies;
    
    int n_samples = X.rows();
    std::vector<int> indices(n_samples);
    for (int i = 0; i < n_samples; ++i) indices[i] = i;

    for (int epoch = 0; epoch < params.epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), gen);
        double epoch_loss = 0.0;
        double epoch_accuracy = 0.0;

        for (int start = 0; start < n_samples; start += params.batch_size) {
            int end = std::min(start + params.batch_size, n_samples);
            int current_batch_size = end - start;

            Eigen::MatrixXd X_batch(current_batch_size, X.cols());
            Eigen::MatrixXd y_batch(current_batch_size, y.cols());
            for (int i = 0; i < current_batch_size; ++i) {
                X_batch.row(i) = X.row(indices[start + i]);
                y_batch.row(i) = y.row(indices[start + i]);
            }

            Eigen::MatrixXd y_pred = forward(X_batch, true);
            backward(X_batch, y_batch);

            epoch_loss += cross_entropy_loss(y_batch, y_pred, weights, params.regularization, params.lambda) * current_batch_size;
            epoch_accuracy += compute_accuracy(y_batch, y_pred) * current_batch_size;
        }

        double loss = epoch_loss / n_samples;
        double train_acc = epoch_accuracy / n_samples;  

        Eigen::MatrixXd y_val_pred = forward(X_val, false);
        double val_loss = cross_entropy_loss(y_val, y_val_pred, weights, params.regularization, params.lambda);
        double val_acc = compute_accuracy(y_val, y_val_pred);

        if (save_accuracies) {
            train_accuracies.push_back(train_acc);
            val_accuracies.push_back(val_acc);
        }
        
        std::cout << "Epoch: " << epoch + 1 
                  << " | Train Loss: " << loss
                  << " | Val Loss: " << val_loss
                  << " | Train Accuracy: " << train_acc * 100 << "%"
                  << " | Val Accuracy: " << val_acc * 100 << "%"  << std::endl;
    }
    
    if (save_accuracies) {
        std::ofstream acc_file("accuracy_log.csv");
        acc_file << "epoch,train_acc,val_acc\n";
        for (size_t i = 0; i < train_accuracies.size(); ++i) {
            acc_file << (i + 1) << "," << train_accuracies[i] << "," << val_accuracies[i] << "\n";
        }
        acc_file.close();
    }
}

Eigen::MatrixXd Network::forward(const Eigen::MatrixXd& X, bool training) {
    activations.clear();
    z_values.clear();
    dropout_masks.clear();

    Eigen::MatrixXd a = X;
    activations.push_back(a);
    std::bernoulli_distribution dropout_dist(1.0 - params.dropout_rate);

    for (size_t i = 0; i < weights.size(); i++) {
        Eigen::MatrixXd z = ((a * weights[i].transpose()).rowwise() + biases[i].transpose());
        z_values.push_back(z);

        if (i == weights.size() - 1) {
            a = softmax(z);
        } else if (params.activation == Activation::ReLU) {
            a = relu(z);
        } else if (params.activation == Activation::Sigmoid) {
            a = sigmoid(z);
        } else if (params.activation == Activation::Tanh) {
            a = tanh(z);
        }

        if (training && params.dropout_rate > 0.0 && i < weights.size() - 1) {
            Eigen::MatrixXd mask = Eigen::MatrixXd::NullaryExpr(a.rows(), a.cols(), 
            [&]() { return dropout_dist(gen) ? 1.0 : 0.0; });

            a = a.array() * mask.array();
            a /= (1.0 - params.dropout_rate);
            dropout_masks.push_back(mask);
        } else {
            dropout_masks.push_back(Eigen::MatrixXd::Ones(a.rows(), a.cols()));
        }
        
        activations.push_back(a);
    }    

    return a;
}

void Network::backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) {
    std::vector<Eigen::MatrixXd> dW(weights.size());
    std::vector<Eigen::VectorXd> db(weights.size());

    // output layer delta (softmax + cross-entropy simplification)
    Eigen::MatrixXd delta = activations.back() - y;

    for (int i = weights.size() - 1; i >= 0; --i) {
        dW[i] = (delta.transpose() * activations[i]) / X.rows();
        db[i] = delta.colwise().mean();

        if (params.regularization == Regularization::L2) {
            dW[i] += (params.lambda / X.rows()) * weights[i];
        }
        else if (params.regularization == Regularization::L1) {
            dW[i] += (params.lambda / X.rows()) * weights[i].array().sign().matrix();
        }

        if (i > 0) {
            Eigen::MatrixXd da = delta * weights[i];
            da.array() *= dropout_masks[i - 1].array();

            Eigen::MatrixXd dz;
            if (params.activation == Activation::ReLU)
                dz = da.array() * relu_derivative(z_values[i - 1]).array();
            else if (params.activation == Activation::Sigmoid)
                dz = da.array() * sigmoid_derivative(z_values[i - 1]).array();
            else if (params.activation == Activation::Tanh)
                dz = da.array() * tanh_derivative(z_values[i - 1]).array();

            delta = dz;
        }
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= params.learning_rate * dW[i];
        biases[i] -= params.learning_rate * db[i];
    }
}

void Network::build_layer_sizes(int input_size, int output_size) {
    layer_sizes.clear();
    layer_sizes.push_back(input_size);
    for (int size : params.hidden_layers) {
        layer_sizes.push_back(size);
    }
    layer_sizes.push_back(output_size);
}

void Network::initialize_weights() {
    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        Eigen::MatrixXd W;
        if (params.init_type == InitType::Xavier) {
            W = xavier_init(layer_sizes[i+1], layer_sizes[i]);
        } else {
            W = he_init(layer_sizes[i+1], layer_sizes[i]);
        }
        weights.push_back(W);
        biases.push_back(Eigen::VectorXd::Zero(layer_sizes[i+1]));
    }
}

double Network::compute_accuracy(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) {
    int correct = 0;
    for (int i = 0; i < y_true.rows(); i++) {
        Eigen::Index pred_label, true_label;
        y_true.row(i).maxCoeff(&true_label);
        y_pred.row(i).maxCoeff(&pred_label);
        if (pred_label == true_label) correct++;
    }
    return static_cast<double>(correct) / y_true.rows();
}