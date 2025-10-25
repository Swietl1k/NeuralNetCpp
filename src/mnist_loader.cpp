#include "mnist_loader.hpp"
#include "types.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>

std::tuple<DynamicMatrix, DynamicMatrix> load_fashion_mnist_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    // skip header
    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV file is empty: " + filename);
    }

    std::vector<int> labels;
    std::vector<std::vector<RealType>> pixels_data;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        if (!std::getline(ss, value, ',')) {
            throw std::runtime_error("Unexpected empty label");
        }
        int label = std::stoi(value);
        labels.push_back(label);

        std::vector<RealType> pixels(784);
        for (int i = 0; i < 784; i++) {
            if (!std::getline(ss, value, ',')) {
                throw std::runtime_error("Unexpected end of row in " + filename);
            }
            pixels[i] = static_cast<RealType>(std::stod(value)) / 255.0f; // normalize to [0,1]
        }

        pixels_data.push_back(std::move(pixels));
    }

    std::cout << "Loaded " << labels.size() << " samples from " << filename << std::endl;

    size_t num_samples = labels.size();
    DynamicMatrix X(num_samples, 784);
    DynamicMatrix Y = DynamicMatrix::Zero(num_samples, 10);

    for (size_t i = 0; i < num_samples; i++) {
        for (int j = 0; j < 784; j++) {
            X(i, j) = pixels_data[i][j];
        }
        Y(i, labels[i]) = 1.0; // one-hot encoding
    }

    return {X, Y};
}
