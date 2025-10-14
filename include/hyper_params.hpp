#pragma once
#include <string>
#include <vector>

enum class InitType { Xavier, He };

enum class Activation { ReLU, Sigmoid, Tanh };

enum class Regularization { None, L2, L1, Dropout };

struct HyperParams {
    std::vector<int> hidden_layers = {64, 32};
    double learning_rate = 0.01;
    int epochs = 50;
    int batch_size = 32;
    double lambda = 0.001;
    double dropout_rate = 0.0;
    Regularization regularization = Regularization::L2;
    InitType init_type = InitType::He;
    Activation activation = Activation::ReLU;
};

std::vector<int> parse_hidden_layers(const std::string& arg);