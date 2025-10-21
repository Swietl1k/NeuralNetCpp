#include "mnist_loader.hpp"
#include "network.hpp"
#include "params.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <string>

int main() {
    std::string train_path = "../mnist/fashion-mnist_train.csv";
    std::string test_path  = "../mnist/fashion-mnist_test.csv";

    // Now loads are dynamic
    auto [X_train_full, y_train_full] = load_fashion_mnist_csv(train_path);
    auto [X_test, y_test]= load_fashion_mnist_csv(test_path);

    DynamicMatrix X_valid = X_train_full.topRows(ValidationSamples);
    DynamicMatrix y_valid = y_train_full.topRows(ValidationSamples);
    DynamicMatrix X_train = X_train_full.bottomRows(X_train_full.rows() - ValidationSamples);
    DynamicMatrix y_train = y_train_full.bottomRows(y_train_full.rows() - ValidationSamples);

    StaticNetwork network;
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Starting training...\n";

    network.train(X_train, y_train, X_valid, y_valid, true);

    auto [test_loss, accuracy] = network.evaluate(X_test, y_test);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Training and evaluation took " << elapsed.count() << " seconds.\n";
    std::cout << "Test Loss: " << test_loss << ", Accuracy: " << accuracy * 100 << "%\n";

    return 0;
}
