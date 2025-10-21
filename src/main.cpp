#include "mnist_loader.hpp"
#include "hyper_params.hpp"
#include "network.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <chrono>


int main(int argc, char* argv[]) {
    HyperParams params;
    if (argc > 1) params.hidden_layers = parse_hidden_layers(argv[1]);
    if (argc > 2) params.learning_rate = std::stod(argv[2]);
    if (argc > 3) params.epochs = std::stoi(argv[3]);
    if (argc > 4) params.batch_size = std::stoi(argv[4]);
    if (argc > 5) params.lambda = std::stod(argv[5]);
    if (argc > 6) params.dropout_rate = std::stod(argv[6]);

    std::string train_data = "../mnist/fashion-mnist_train.csv";
    std::string test_data = "../mnist/fashion-mnist_test.csv";

    auto [X_train, y_train] = load_fashion_mnist_csv(train_data);
    auto [X_test, y_test] = load_fashion_mnist_csv(test_data);

    int n_val = X_train.rows() * 0.1;
    Eigen::MatrixXd X_valid = X_train.topRows(n_val);
    Eigen::MatrixXd y_valid = y_train.topRows(n_val);
    Eigen::MatrixXd X_train_wo_val = X_train.bottomRows(X_train.rows() - n_val);
    Eigen::MatrixXd y_train_wo_val = y_train.bottomRows(y_train.rows() - n_val);
 
    Network network = Network(784, 10, params);
    auto start = std::chrono::high_resolution_clock::now();

    network.train(X_train_wo_val, X_valid, y_train_wo_val, y_valid, true);

    auto [test_loss, accuracy] = network.evaluate(X_test, y_test);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Training and evaluation took " << elapsed.count() << " seconds." << std::endl;

    std::cout << "Test Loss: " << test_loss << ", Accuracy: " << accuracy * 100 << "%" << std::endl;

    return 0;
}