#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <matrix.h>
#include <network.h>
#include <networkFast.h>
//ChatGPT kode for å gjøre mnist data om til vectorform som kan enkelt gjøres om til matriser


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