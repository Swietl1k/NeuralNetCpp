#include "params.hpp"
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
            throw std::runtime_error("Unexpected empty label in " + filename);
        }

        int label = std::stoi(value);
        labels.push_back(label);

        std::vector<RealType> pixels(InputSize);
        for (int i = 0; i < InputSize; i++) {
            if (!std::getline(ss, value, ',')) {
                throw std::runtime_error("Unexpected end of row in " + filename);
            }
            pixels[i] = std::stof(value) / static_cast<RealType>(255.0);
            //pixels[i] = static_cast<RealType>(std::stoi(value) / static_cast<RealType>(255.0)); // normalize to [0,1]
        }
        pixels_data.push_back(std::move(pixels));
    }

    const size_t num_samples = labels.size();
    std::cout << "Loaded " << num_samples << " samples from " << filename << std::endl;

    DynamicMatrix X(num_samples, InputSize);
    DynamicMatrix Y = DynamicMatrix::Zero(num_samples, OutputSize);

    for (size_t i = 0; i < num_samples; ++i) {
        for (int j = 0; j < InputSize; ++j) {
            X(static_cast<int>(i), j) = pixels_data[i][j];
        }
        Y(static_cast<int>(i), labels[i]) = 1.0f; // one-hot encoding
    }

    return {X, Y};
}
