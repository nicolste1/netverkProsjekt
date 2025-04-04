#include <network.h>
#include <math.h> //Sigmoid
#include <algorithm> // shuffle vector, 
#include <random>


//Konstruktør og destruktør_____________________________________________________________________________________

Network::Network(std::vector<int> inNeurovec, Matrix inInputLayer) : neurovec(inNeurovec), inputLayer(inInputLayer)
{
    if (inputLayer.getColums() != 1) {
        throw std::invalid_argument("inputLayer matrise er ikke en gyldig vektor");
    }
    numLayers = neurovec.size();
    Matrix(neurovec[numLayers-1],1);
    outLayer = Matrix(neurovec[numLayers-1],1);
    outLayer.setValue(0);
    //Lage random matriser
    //lage random biaser
    for(int i = 1; i < numLayers; i++){
        Matrix addLayer(neurovec.at(i), neurovec.at(i-1));
        addLayer.setRandomValues();
        layers.push_back(addLayer);

        Matrix addBias(neurovec.at(i),1);
        addBias.setRandomValues();
        biases.push_back(addBias);
    }

}

Network::~Network(){

}


//medlemsfunksjoner___________________________________________________________________________________________

//feedforward tar inn inputlayer og gir ut outputlayer 
void Network::feedforward(){
    for (int i = 0; i < numLayers - 1; i++){
        inputLayer = layers[i] * inputLayer + biases[i];
        inputLayer.applyActivationFunc("sigmoid");
    }
    outLayer = inputLayer;
}

void Network::applySGD(std::vector <std::tuple<Matrix, Matrix>> trainData, int numEpocks, int miniBatchSize, double learnRate, std::vector <std::tuple<Matrix, Matrix>> testData){
    for(int i = 0; i < numEpocks; i++){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(trainData.begin(), trainData.end(),gen);
        for(int j = 0; j < trainData.size()/miniBatchSize; j++){
            std::vector <std::tuple<Matrix, Matrix>> miniBatch(trainData.begin() + j * miniBatchSize,trainData.begin() + (j + 1 ) * miniBatchSize);
            updateMiniBatch(miniBatch, learnRate);
        }
        //Print epock, og feil
        std::cout << "epoch: " << i;
        if(!testData.empty()){
            evaluate(testData);
        }
    }
}

void Network::updateMiniBatch(std::vector<std::tuple<Matrix,Matrix>> miniBatch,double learnRate){
    //Lag korosponderende tom biases og layers som representerer gradient c
    std::vector<Matrix> gradientLayers = makeEmptyLayers();
    std::vector<Matrix> gradientBiases = makeEmptyBiases();

    //For hvert treningseksempel kjør backprop og opptater gradient c
    for(std::tuple<Matrix,Matrix> example: miniBatch){
        backProp2(example, gradientLayers,gradientBiases);
    }
    //Opptater biases og layers med gradient
    for(int i = 0; i < numLayers - 1; i++){
        //std::cout << "layers at " << i << std::endl << this->layers.at(i)  << "gradienlayers at " << i << std::endl << gradientLayers.at(i) * (learnRate/miniBatch.size()) << std::endl;
        this->layers.at(i) = this->layers.at(i) - gradientLayers.at(i) * (learnRate/miniBatch.size());
        this->biases.at(i) = this->biases.at(i) - gradientBiases.at(i) * (learnRate/miniBatch.size());
    }
}

void Network::backProp(std::tuple<Matrix,Matrix> example,std::vector<Matrix>& gradientLayers, std::vector<Matrix>& gradientBiases)
{
    //opptater gradientLayers og Biases
    std::vector<Matrix> deltaGradientLayers = makeEmptyLayers();
    std::vector<Matrix> deltaGradientBiases = makeEmptyBiases();
    Matrix a1 = std::get<0>(example);
    std::vector<Matrix> activations = {a1};
    Matrix emtyMatrix;
    std::vector<Matrix> z = {emtyMatrix};
    Matrix outError; 
    //feedforward 
    for(int layer = 1; layer < numLayers; layer++){
        Matrix zl = layers.at(layer) * activations.at(layer-1) + biases.at(layer);
        z.push_back(zl);
        zl.applyActivationFunc("sigmoid");
        activations.push_back(zl);
    }
    //backProp
    Matrix dcOut = activations.at(numLayers-1) - std::get<1>(example);
    Matrix sigmoidPrime = z.at(numLayers-1);
    sigmoidPrime.sigmoidPrime(); //Mulighet for å bruke activations.at rett inn i sigPrime
    outError = (dcOut.hademart(sigmoidPrime));
    gradientBiases.at(numLayers-2) = gradientBiases.at(numLayers-2) + outError; 
    gradientLayers.at(numLayers-2) = gradientLayers.at(numLayers-2) + activations.at(numLayers - 2) * outError;
    for(int layer = numLayers-3; layer >= 0; layer--){
        Matrix sigmoidPrime = z.at(layer + 1);
        outError = (layers.at(layer + 1).transpose() * outError).hademart(sigmoidPrime);
        gradientBiases.at(layer) = gradientBiases.at(layer) + outError;
        gradientLayers.at(layer) = gradientLayers.at(layer) + activations.at(layer) * outError;
    } 
}

void Network::backProp2(std::tuple<Matrix,Matrix> example,std::vector<Matrix>& gradientLayers, std::vector<Matrix>& gradientBiases){
    //opptater gradientLayers og Biases
    std::vector<Matrix> deltaGradientLayers = makeEmptyLayers();
    std::vector<Matrix> deltaGradientBiases = makeEmptyBiases();
    Matrix a1 = std::get<0>(example);
    std::vector<Matrix> activations = {a1};
    std::vector<Matrix> z;  
    //feedforward 
    for (int layer = 1; layer < numLayers; ++layer) {
        Matrix zl = layers.at(layer - 1) * activations.at(layer - 1) + biases.at(layer - 1);
        z.push_back(zl);
        zl.applyActivationFunc("sigmoid");
        activations.push_back(zl);
    }
    //backProp
    Matrix dcOut = activations.back() - std::get<1>(example);
    Matrix sigmoidPrime = z.back();
    sigmoidPrime.sigmoidPrime();
    Matrix outError = dcOut.hademart(sigmoidPrime);
    gradientBiases.at(numLayers - 2) = gradientBiases.at(numLayers - 2) + outError;
    gradientLayers.at(numLayers - 2) = gradientLayers.at(numLayers - 2) + (outError * activations.at(numLayers - 2).transpose());
    for (int layer = numLayers - 3; layer >= 0; --layer) {
        Matrix sp = z.at(layer);
        sp.sigmoidPrime();
        outError = (layers.at(layer + 1).transpose() * outError).hademart(sp);
        gradientBiases.at(layer) = gradientBiases.at(layer) + outError;
        gradientLayers.at(layer) = gradientLayers.at(layer) + outError * activations.at(layer).transpose();
    }
}

std::vector<Matrix> Network::makeEmptyLayers(){
    std::vector<Matrix> zeroLayers;
    for(int i = 1; i < numLayers; i++){
        Matrix addLayer(neurovec.at(i), neurovec.at(i-1));
        addLayer.setValue(0);
        zeroLayers.push_back(addLayer);
    }
    return zeroLayers;
}
std::vector<Matrix> Network::makeEmptyBiases(){
    std::vector<Matrix> zeroBiases;
    for(int i = 1; i < numLayers; i++){
        Matrix addBias(neurovec.at(i),1);
        addBias.setValue(0);
        zeroBiases.push_back(addBias);
    }
    return zeroBiases;
}


void Network::evaluate(std::vector<std::tuple<Matrix,Matrix>> testData){
    int numRight = 0;
    for(std::tuple<Matrix,Matrix> test : testData){
        inputLayer = std::get<0>(test);
        feedforward();
        int predicted = outLayer.argMax();
        int actual = std::get<1>(test).argMax();
        if(predicted == actual)
        {
            numRight++;
        }
        //std::cout << "predicted:  " << std::endl << outLayer << "actual: " << std::get<1>(test) << std::endl;   
    }
    std::cout << " " << numRight << " riktige" << std::endl;
}