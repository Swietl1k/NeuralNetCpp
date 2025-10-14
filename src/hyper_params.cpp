#include "hyper_params.hpp"
#include <sstream>
#include <string>
#include <cstdlib>

std::vector<int> parse_hidden_layers(const std::string& arg) {
    std::vector<int> layers;
    std::stringstream ss(arg);
    std::string item;
    while (std::getline(ss, item, ',')) {
        layers.push_back(std::stoi(item));
    }
    return layers;
}