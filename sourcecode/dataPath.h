#pragma once 
#include <string>
#include <filesystem>
#include <iostream>

//MNIST loadDATA
inline const std::string TrainImages = "data/train-images-idx3-ubyte";
inline const std::string TrainLabels = "data/train-labels-idx1-ubyte";
inline const std::string TestImages = "data/t10k-images-idx3-ubyte";
inline const std::string TestLabels = "data/t10k-labels-idx1-ubyte";

inline const std::string dataPath = "data/";

std::string& fileExist(std::string& filePath);