#pragma once
#include <matrix.h>
#include <vector>
#include <tuple>

class Network{
    private:
        std::vector<int> neurovec; //Vector som innholder antall layers og resperktivt antall neurons i hvert layer
        //feks [3,2,3] har 3 layers med 3 neurons i første og 2 i andre. 
        Matrix inputLayer;
        Matrix outLayer;
        int numLayers;
        std::vector<Matrix> layers;
        std::vector<Matrix> biases;
        std::vector<Matrix> makeEmptyLayers();
        std::vector<Matrix> makeEmptyBiases();
        void updateMiniBatch(std::vector<std::tuple<Matrix,Matrix>> miniBatch,double learnRate);
        void backProp(std::tuple<Matrix,Matrix> example,std::vector<Matrix>& gradientLayers, std::vector<Matrix>& gradientBiases);
        void backProp2(std::tuple<Matrix,Matrix> example,std::vector<Matrix>& gradientLayers, std::vector<Matrix>& gradientBiases);
    public:
        Matrix getInputLayer(){ return inputLayer; }
        Matrix getOutLayer(){ return outLayer; }
        Network(std::vector<int> inNeurovec, Matrix inputLayer);
        ~Network();
        void applySGD(std::vector <std::tuple<Matrix, Matrix>> trainData, int numEpocks, int miniBatchSize, double learnRate, std::vector <std::tuple<Matrix, Matrix>> testData); //stochastic gradient decent 
        void feedforward();
        void evaluate(std::vector<std::tuple<Matrix,Matrix>> testData);
};

//hjelpefunksjoner