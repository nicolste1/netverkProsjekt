#include "mnist_loader.hpp"
#include <fstream>
#include <iostream>

uint32_t readBigEndianInt(std::ifstream& f) {
    unsigned char bytes[4];
    f.read((char*)bytes, 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

MNISTData load_mnist_images_and_labels(const std::string& image_file, const std::string& label_file) {
    std::ifstream imageStream(image_file, std::ios::binary);
    std::ifstream labelStream(label_file, std::ios::binary);

    if (!imageStream || !labelStream) {
        throw std::runtime_error("Klarte ikke åpne MNIST-filer.");
    }

    MNISTData data;

    // Header for bilder
    uint32_t magic_images = readBigEndianInt(imageStream);
    uint32_t num_images = readBigEndianInt(imageStream);
    uint32_t rows = readBigEndianInt(imageStream);
    uint32_t cols = readBigEndianInt(imageStream);

    data.image_rows = rows;
    data.image_cols = cols;

    // Header for labels
    uint32_t magic_labels = readBigEndianInt(labelStream);
    uint32_t num_labels = readBigEndianInt(labelStream);

    if (num_images != num_labels) {
        throw std::runtime_error("Antall bilder og labels stemmer ikke overens.");
    }

    // Lese bilder
    for (uint32_t i = 0; i < num_images; ++i) {
        std::vector<double> img(rows * cols);
        for (uint32_t j = 0; j < rows * cols; ++j) {
            unsigned char pixel;
            imageStream.read((char*)&pixel, 1);
            img[j] = pixel / 255.0;  // Normaliser
        }
        data.images.push_back(img);
    }

    // Lese labels
    for (uint32_t i = 0; i < num_labels; ++i) {
        unsigned char label;
        labelStream.read((char*)&label, 1);
        data.labels.push_back(label);
    }

    return data;
}

void runMNIST(std::string name) {
    if(name == "network"){
        MNISTData train = load_mnist_images_and_labels("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
        MNISTData test = load_mnist_images_and_labels("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

        std::cout << "MNIST lastet: " << train.images.size() << " treningsbilder\n";

        auto train_labels_onehot = one_hot_encode_labels(train.labels, 10);
        auto test_labels_onehot = one_hot_encode_labels(test.labels, 10);
        auto trainData = toMatrixDataset(train.images, train_labels_onehot, 100);
        auto testData  = toMatrixDataset(test.images, test_labels_onehot, 10);
        std::vector<int> inputVector = {784,30,10};
        Matrix inputLayer = std::get<0> (trainData.at(0));
        Network net(inputVector,inputLayer);
        std::cout << "before sgd: " << std::endl;
        net.feedforward();
        std::cout << net.getOutLayer();
        net.applySGD(trainData, 30, 5, 3.0, testData);
    }
    else if(name == "networkFast"){
        MNISTData train = load_mnist_images_and_labels("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
        MNISTData test = load_mnist_images_and_labels("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

        std::cout << "MNIST lastet: " << train.images.size() << " treningsbilder\n";

        auto train_labels_onehot = one_hot_encode_labels(train.labels, 10);
        auto test_labels_onehot = one_hot_encode_labels(test.labels, 10);
        auto trainData = toMatrix2Dataset(train.images, train_labels_onehot, 60000);
        auto testData  = toMatrix2Dataset(test.images, test_labels_onehot, 10000);
        std::vector<int> inputVector = {784,24,12,10};
        Matrix2 inputLayer = std::get<0> (trainData.at(0));
        NetworkFast net(inputVector,inputLayer);
        std::cout << " outlayer before sgd: " << std::endl;
        net.feedforward();
        std::cout << net.getOutLayer();
        net.applySGD(trainData, 10, 10, 3.0, testData);
        std::cout << " outlayer after sgd: " << std::endl;
        std::cout << net.getOutLayer();
    }
    else{
        std::cout<< "Du må gi inn navnet på hvilket netverk du vil bruke (network) eller (networkFast) " << std::endl;
    }
}

std::vector<std::vector<double>> one_hot_encode_labels(const std::vector<uint8_t>& labels, int num_classes = 10) {
    std::vector<std::vector<double>> encoded;
    for (uint8_t label : labels) {
        std::vector<double> vec(num_classes, 0.0);
        vec[label] = 1.0;
        encoded.push_back(vec);
    }
    return encoded;
}

std::vector<std::tuple<Matrix, Matrix>> toMatrixDataset(
    const std::vector<std::vector<double>>& images,
    const std::vector<std::vector<double>>& labels,
    int antPic
) {
    if(antPic > images.size()){
        antPic = images.size();
        std::cout << "Sendt inn antPic større en imgages.size(), antPic er nå lik størelsen på images" << std::endl;
    }
    std::vector<std::tuple<Matrix, Matrix>> dataset;
    for (size_t i = 0; i < antPic; ++i) {
        dataset.emplace_back(Matrix(images[i]), Matrix(labels[i]));
    }
    return dataset;
}

std::vector<std::tuple<Matrix2, Matrix2>> toMatrix2Dataset(
    const std::vector<std::vector<double>>& images,
    const std::vector<std::vector<double>>& labels,
    int antPic
) {
    if(antPic > images.size()){
        antPic = images.size();
        std::cout << "Sendt inn antPic større en imgages.size(), antPic er nå lik størelsen på images" << std::endl;
    }
    std::vector<std::tuple<Matrix2, Matrix2>> dataset;
    for (size_t i = 0; i < antPic; ++i) {
        dataset.emplace_back(Matrix2(images[i]), Matrix2(labels[i]));
    }
    return dataset;
}