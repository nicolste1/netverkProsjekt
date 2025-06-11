#pragma once
#include <vector>
#include <string>
#include <cstdint>

#include "slowNetwork/network.h"
#include "fastNetwork/networkFast.h"

extern const int PixelwidthDraw;
extern const int PixelheightDraw;
extern const int sideBarWidth;
extern const int sideBarPadding;
extern const int heightPadding;
extern const int titleBarHeight;
extern const int fontTitle;
extern const int fontGuess;
extern const int width;
extern const int height;
extern const int MNISTsize;

int argMax(const std::vector<double>& vec);
std::vector<std::tuple<std::vector<double>, double>> returnTestVector(int antPic);

 
struct MNISTData {
    std::vector<std::vector<double>> images;
    std::vector<uint8_t> labels;
    int image_rows;
    int image_cols;
};

MNISTData load_mnist_images_and_labels(const std::string& image_file, const std::string& label_file);
void runMNIST(std::string name);  // funksjon som kjører test

std::vector<std::vector<double>> one_hot_encode_labels(const std::vector<uint8_t>& labels, int num_classes);

std::vector<std::tuple<Matrix, Matrix>> toMatrixDataset(
    const std::vector<std::vector<double>>& images,
    const std::vector<std::vector<double>>& labels,
    int antPic

);

std::vector<std::tuple<Matrix2, Matrix2>> toMatrix2Dataset(
    const std::vector<std::vector<double>>& images,
    const std::vector<std::vector<double>>& labels,
    int antPic
);

void updatePersonalNet(NetworkFast &net, std::vector<std::tuple<std::vector<double>, double>> PersonalTrainData, int numEpocks = 1, int miniBatchSize = 10, double learnRate = 1.0);

void oversample(std::vector<std::tuple<std::vector<double>, double>>& vec, int numDuplicate);


std::vector<double> moveLeftRight(const std::vector<double>& inVec, std::string moveDir);
bool checkIfColumClear(const std::vector<double>& inVec, int colum);