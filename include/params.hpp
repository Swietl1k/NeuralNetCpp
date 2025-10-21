#pragma once

#include <Eigen/Dense>

using RealType = float;

template<int Rows, int Cols>
using MatrixT = Eigen::Matrix<RealType, Rows, Cols>;

template<int Size>
using VectorT = Eigen::Matrix<RealType, Size, 1>;

using DynamicMatrix = Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>;
using DynamicVector = Eigen::Matrix<RealType, Eigen::Dynamic, 1>;


enum class InitType { Xavier, He };

enum class Activation { ReLU, Sigmoid, Tanh };

enum class Regularization { None, L2, L1, Dropout };


inline constexpr int InputSize = 784;
inline constexpr int OutputSize = 10;
inline constexpr int BatchSize = 32;
inline constexpr int TrainSamples = 60000; 
inline constexpr int TestSamples = 10000;
inline constexpr int ValidationSamples = 6000;
inline constexpr int L1 = 128;
inline constexpr int L2 = 64;
inline constexpr int L3 = 32;
inline constexpr int Epochs = 100;
inline constexpr RealType LearningRate = 0.01;
inline constexpr RealType Lambda = 0.003;
inline constexpr RealType DropoutRate = 0.17;
inline constexpr Regularization Reg = Regularization::L2;
inline constexpr InitType Init = InitType::He;
inline constexpr Activation Act = Activation::ReLU;

