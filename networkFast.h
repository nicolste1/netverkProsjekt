#pragma once
#include <matrix2.h>
#include <vector>
#include <tuple>
#include <fstream>

class NetworkFast{
    private:
        std::vector<int> neurovec; //Vector som innholder antall layers og resperktivt antall neurons i hvert layer
        //feks [3,2,3] har 3 layers med 3 neurons i første og 2 i andre. 
        Matrix2 inputLayer;
        Matrix2 outLayer;
        int numLayers;
        std::vector<Matrix2> layers;
        std::vector<Matrix2> biases;
        std::vector<Matrix2> makeEmptyLayers();
        std::vector<Matrix2> makeEmptyBiases();
        void updateMiniBatch(std::vector<std::tuple<Matrix2,Matrix2>> miniBatch,double learnRate);
        void backProp(std::tuple<Matrix2,Matrix2> example,std::vector<Matrix2>& gradientLayers, std::vector<Matrix2>& gradientBiases);
        void backProp2(std::tuple<Matrix2,Matrix2> example,std::vector<Matrix2>& gradientLayers, std::vector<Matrix2>& gradientBiases);
    public:
        Matrix2 getInputLayer(){ return inputLayer; }
        Matrix2 getOutLayer(){ return outLayer; }
        NetworkFast(std::vector<int> inNeurovec, Matrix2 inputLayer);
        NetworkFast(std::vector<int> inNeurovec);
        NetworkFast(const std::string& filename);
        void applySGD(std::vector <std::tuple<Matrix2, Matrix2>> trainData, int numEpocks, int miniBatchSize, double learnRate, std::vector <std::tuple<Matrix2, Matrix2>> testData); //stochastic gradient decent 
        void feedforward();
        int feedforward(std::vector<double> inVec);
        void evaluate(std::vector<std::tuple<Matrix2,Matrix2>> testData);
        void saveNetworkToFile(const std::string& filename);
        void printNetworkToTerminal();
};

//hjelpefunksjoner
template<typename T> 
std::ostream& operator <<(std::ostream& os, const std::vector<T>& vec){
    for(size_t i = 0; i < vec.size(); i++){
        os <<vec.at(i);
        os << std::endl;
    }
    return os;
}


template<typename T> 
std::istream& operator >>(std::istream& is, std::vector<T>& vec){
    std::string element;
    while(true){
        is >> element;
        if(std::isdigit(element[0])){
            vec.push_back(std::stod(element));
        }
        else{
            break;
        }
    }
    return is;
}


