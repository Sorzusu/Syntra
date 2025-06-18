#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>

enum class ActivationType { Sigmoid, ReLU, Tanh };

ActivationType parseActivation(const std::string& str) {
    if (str == "sigmoid") return ActivationType::Sigmoid;
    if (str == "relu") return ActivationType::ReLU;
    if (str == "tanh") return ActivationType::Tanh;
    throw std::invalid_argument("Unknown activation function: " + str);
}

double activate(double x, ActivationType type) {
    switch (type) {
        case ActivationType::Sigmoid: return 1.0 / (1.0 + std::exp(-x));
        case ActivationType::ReLU: return std::max(0.0, x);
        case ActivationType::Tanh: return std::tanh(x);
        default: return x;
    }
}

double activateDerivative(double x, ActivationType type) {
    switch (type) {
        case ActivationType::Sigmoid: {
            double s = activate(x, ActivationType::Sigmoid);
            return s * (1.0 - s);
        }
        case ActivationType::ReLU: return x > 0.0 ? 1.0 : 0.0;
        case ActivationType::Tanh: {
            double t = activate(x, ActivationType::Tanh);
            return 1.0 - t * t;
        }
        default: return 1.0;
    }
}

std::mt19937 rng(std::random_device{}());

class NeuralNetwork {
private:
    std::vector<std::vector<double>> layers;
    std::vector<std::vector<std::vector<double>>> weights;
    ActivationType activation;

public:
    NeuralNetwork(const std::vector<int>& layerSizes, ActivationType type) : activation(type) {
        int numLayers = layerSizes.size();
        layers.resize(numLayers);
        weights.resize(numLayers - 1);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < numLayers; ++i) {
            layers[i].resize(layerSizes[i]);
        }
        for (int i = 0; i < numLayers - 1; ++i) {
            int inputSize = layerSizes[i];
            int outputSize = layerSizes[i + 1];
            weights[i].resize(inputSize, std::vector<double>(outputSize));
            for (auto& row : weights[i]) {
                for (auto& w : row) {
                    w = dist(rng);
                }
            }
        }
    }

    std::vector<double> feedForward(const std::vector<double>& input) {
        layers[0] = input;
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < layers[i + 1].size(); ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < layers[i].size(); ++k) {
                    sum += layers[i][k] * weights[i][k][j];
                }
                layers[i + 1][j] = activate(sum, activation);
            }
        }
        return layers.back();
    }

    void train(const std::vector<std::vector<double>>& trainData,
               const std::vector<std::vector<double>>& targets,
               int epochs, double lr, int batchSize, double reg,
               const std::string& logFile,
               const std::vector<std::vector<double>>& valData,
               const std::vector<std::vector<double>>& valTargets,
               double earlyStop) {

        std::ofstream log(logFile);
        if (!log) throw std::runtime_error("Can't open log file.");

        std::vector<int> indices(trainData.size());
        std::iota(indices.begin(), indices.end(), 0);

        double bestError = std::numeric_limits<double>::max();
        int bestEpoch = 0;
        auto bestWeights = weights;

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);
            double epochError = 0.0;

            for (int i = 0; i < static_cast<int>(trainData.size()); i += batchSize) {
                auto batchEnd = std::min(i + batchSize, static_cast<int>(trainData.size()));

                std::vector<std::vector<std::vector<double>>> grad(weights.size());
                for (size_t l = 0; l < weights.size(); ++l) {
                    grad[l].resize(weights[l].size(), std::vector<double>(weights[l][0].size(), 0.0));
                }

                for (int j = i; j < batchEnd; ++j) {
                    int idx = indices[j];
                    std::vector<std::vector<double>> localOutput = layers;
                    localOutput[0] = trainData[idx];

                    for (size_t l = 0; l < weights.size(); ++l) {
                        for (size_t n = 0; n < weights[l][0].size(); ++n) {
                            double sum = 0.0;
                            for (size_t m = 0; m < weights[l].size(); ++m) {
                                sum += localOutput[l][m] * weights[l][m][n];
                            }
                            localOutput[l + 1][n] = activate(sum, activation);
                        }
                    }

                    std::vector<std::vector<double>> deltas(layers.size());
                    deltas.back().resize(localOutput.back().size());

                    for (size_t k = 0; k < targets[idx].size(); ++k) {
                        double err = targets[idx][k] - localOutput.back()[k];
                        deltas.back()[k] = err;
                        epochError += err * err;
                    }

                    for (int l = static_cast<int>(weights.size()) - 1; l >= 0; --l) {
                        deltas[l].resize(layers[l].size());
                        for (size_t i = 0; i < weights[l].size(); ++i) {
                            for (size_t j = 0; j < weights[l][i].size(); ++j) {
                                grad[l][i][j] += deltas[l + 1][j] * localOutput[l][i];
                            }
                        }
                    }
                }

                for (size_t l = 0; l < weights.size(); ++l) {
                    for (size_t i = 0; i < weights[l].size(); ++i) {
                        for (size_t j = 0; j < weights[l][i].size(); ++j) {
                            weights[l][i][j] += (lr * grad[l][i][j] / batchSize) - (reg * weights[l][i][j]);
                        }
                    }
                }
            }

            double valError = 0.0;
            for (size_t i = 0; i < valData.size(); ++i) {
                auto out = feedForward(valData[i]);
                for (size_t j = 0; j < out.size(); ++j) {
                    valError += std::pow(valTargets[i][j] - out[j], 2);
                }
            }
            valError /= (2 * valData.size());

            log << "Epoch " << epoch << " Training Error: " << epochError / trainData.size()
                << " Validation Error: " << valError << std::endl;

            if (valError < bestError) {
                bestError = valError;
                bestEpoch = epoch;
                bestWeights = weights;
            } else if (epoch - bestEpoch >= earlyStop) {
                log << "Early stopping at epoch " << epoch << std::endl;
                break;
            }
        }

        weights = bestWeights;
        log << "Best epoch: " << bestEpoch << std::endl;
        log.close();
    }

    void saveModel(const std::string& path) {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("Can't open save file.");
        f << layers.size() << "\n";
        for (const auto& l : layers) f << l.size() << " ";
        f << "\n";
        for (const auto& lw : weights) {
            for (const auto& nw : lw) {
                for (double w : nw) f << w << " ";
            }
            f << "\n";
        }
    }

    void loadModel(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Can't open model file.");
        int nLayers; f >> nLayers;
        layers.resize(nLayers);
        for (int i = 0; i < nLayers; ++i) {
            int sz; f >> sz; layers[i].resize(sz);
        }
        weights.resize(nLayers - 1);
        for (auto& lw : weights) {
            lw.resize(layers[&lw - &weights[0]].size());
            for (auto& nw : lw) {
                nw.resize(layers[&lw - &weights[0] + 1].size());
                for (auto& w : nw) f >> w;
            }
        }
    }
};

std::vector<std::vector<double>> loadCSV(const std::string& path, int& inputSize, int& outputSize) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Can't open CSV file.");
    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string val;
        std::vector<double> row;
        while (std::getline(ss, val, ',')) {
            row.push_back(std::stod(val));
        }
        data.push_back(row);
    }
    inputSize = data[0].size() - 1;
    outputSize = 1;
    return data;
}

int main(int argc, char* argv[]) {
    std::string csvPath = "", modelSavePath = "model.txt", logFile = "training_log.txt";
    int epochs = 1000, batchSize = 4;
    double lr = 0.1, reg = 0.0, earlyStop = 20;
    ActivationType activation = ActivationType::Sigmoid;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--epochs") epochs = std::stoi(argv[++i]);
        else if (arg == "--lr") lr = std::stod(argv[++i]);
        else if (arg == "--batch") batchSize = std::stoi(argv[++i]);
        else if (arg == "--activation") activation = parseActivation(argv[++i]);
        else if (arg == "--earlystop") earlyStop = std::stod(argv[++i]);
        else if (arg == "--save") modelSavePath = argv[++i];
        else if (arg == "--log") logFile = argv[++i];
        else csvPath = arg;
    }

    std::vector<std::vector<double>> trainingData;
    std::vector<std::vector<double>> targetOutputs;

    if (!csvPath.empty()) {
        int inputSize, outputSize;
        auto csvData = loadCSV(csvPath, inputSize, outputSize);
        for (const auto& row : csvData) {
            trainingData.emplace_back(row.begin(), row.begin() + inputSize);
            targetOutputs.emplace_back(row.begin() + inputSize, row.end());
        }
    } else {
        trainingData = {{0,0},{0,1},{1,0},{1,1}};
        targetOutputs = {{0},{1},{1},{0}};
    }

    auto validationData = trainingData;
    auto validationTargets = targetOutputs;

    NeuralNetwork nn({2, 4, 1}, activation);
    nn.train(trainingData, targetOutputs, epochs, lr, batchSize, reg, logFile, validationData, validationTargets, earlyStop);
    nn.saveModel(modelSavePath);
    nn.loadModel(modelSavePath);
    return 0;
}

