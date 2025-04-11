#include <networkFast.h>
#include <math.h> //Sigmoid
#include <algorithm> // shuffle vector, 
#include <random>
//Denne koden bygger på network klassen og dens tilhørende matrixklasse 
//derav kan strukturen til tider oppleves litt uhensiktsmessig 
 
//Mye forbedringspotensiale feks inputLayer er unødvendig og 
//sjekke const corectnes og bruke referanse der mulig 

//Konstruktør og destruktør_____________________________________________________________________________________

NetworkFast::NetworkFast(std::vector<int> inNeurovec, Matrix2 inInputLayer) : neurovec(inNeurovec), inputLayer(inInputLayer)
{
    if (inputLayer.getColums() != 1) {
        throw std::invalid_argument("inputLayer matrise er ikke en gyldig vektor");
    }
    numLayers = neurovec.size();
    Matrix2(neurovec[numLayers-1],1);
    outLayer = Matrix2(neurovec[numLayers-1],1);
    outLayer.setValue(0);
    //Lage random matriser
    //lage random biaser
    for(int i = 1; i < numLayers; i++){
        Matrix2 addLayer(neurovec.at(i), neurovec.at(i-1));
        addLayer.setXavierValues(neurovec.at(i-1), neurovec.at(i));
        layers.push_back(addLayer);

        Matrix2 addBias(neurovec.at(i),1);
        addBias.setRandomValues();
        biases.push_back(addBias);
    }

}

NetworkFast::NetworkFast(std::vector<int> inNeurovec): neurovec(inNeurovec)
{
    numLayers = neurovec.size();
    inputLayer = Matrix2(neurovec.at(0),1);
    Matrix2(neurovec[numLayers-1],1);
    outLayer = Matrix2(neurovec[numLayers-1],1);
    //Lage random matriser
    //lage random biaser
    for(int i = 1; i < numLayers; i++){
        Matrix2 addLayer(neurovec.at(i), neurovec.at(i-1));
        addLayer.setXavierValues(neurovec.at(i-1), neurovec.at(i));
        layers.push_back(addLayer);

        Matrix2 addBias(neurovec.at(i),1);
        addBias.setRandomValues();
        biases.push_back(addBias);
    }

}


NetworkFast::NetworkFast(const std::string& filename){
    std::ifstream inFile(filename);
    if (!inFile) {
        throw std::runtime_error("Kunne ikke åpne fil: " + filename);
    }
    std::string forsteLinje;
    inFile >> forsteLinje;
    inFile >> neurovec;

    numLayers = neurovec.size();
    inputLayer = Matrix2(neurovec.at(0),1);
    Matrix2(neurovec[numLayers-1],1);
    outLayer = Matrix2(neurovec.at(numLayers-1),1);
    
    for(int i = 1; i < numLayers; i++){
        Matrix2 addLayer(neurovec.at(i), neurovec.at(i-1));
        inFile >> addLayer;
        layers.push_back(addLayer);
    }

    for(int i = 1; i < numLayers; i++){
        Matrix2 addBias(neurovec.at(i),1);
        inFile >> addBias;
        biases.push_back(addBias);
    }
}

    



//medlemsfunksjoner___________________________________________________________________________________________

//feedforward tar inn inputlayer og gir ut outputlayer 
void NetworkFast::feedforward(){
    for (int i = 0; i < numLayers - 1; i++){
        inputLayer = layers[i] * inputLayer + biases[i];
        inputLayer.applyActivationFunc("sigmoid");
    }
    outLayer = inputLayer;
}

int NetworkFast::feedforward(std::vector<double> inVec){
    inputLayer = Matrix2(inVec);
    for (int i = 0; i < numLayers - 1; i++){
        inputLayer = layers[i] * inputLayer + biases[i];
        inputLayer.applyActivationFunc("sigmoid");
    }
    return inputLayer.argMax();
}

void NetworkFast::applySGD(std::vector <std::tuple<Matrix2, Matrix2>> trainData, int numEpocks, int miniBatchSize, double learnRate, std::vector <std::tuple<Matrix2, Matrix2>> testData){
    for(int i = 0; i < numEpocks; i++){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(trainData.begin(), trainData.end(),gen);
        for(int j = 0; j < trainData.size()/miniBatchSize; j++){
            std::vector <std::tuple<Matrix2, Matrix2>> miniBatch(trainData.begin() + j * miniBatchSize,trainData.begin() + (j + 1 ) * miniBatchSize);
            updateMiniBatch(miniBatch, learnRate);
        }
        //Print epock, og feil
        std::cout << "epoch: " << i;
        if(!testData.empty()){
            evaluate(testData);
        }
    }
}

void NetworkFast::updateMiniBatch(std::vector<std::tuple<Matrix2,Matrix2>> miniBatch,double learnRate){
    //Lag korosponderende tom biases og layers som representerer gradient c
    std::vector<Matrix2> gradientLayers = makeEmptyLayers();
    std::vector<Matrix2> gradientBiases = makeEmptyBiases();

    //For hvert treningseksempel kjør backprop og opptater gradient c
    for(std::tuple<Matrix2,Matrix2> example: miniBatch){
        backProp2(example, gradientLayers,gradientBiases);
    }
    //Opptater biases og layers med gradient
    for(int i = 0; i < numLayers - 1; i++){
        //std::cout << "layers at " << i << std::endl << this->layers.at(i)  << "gradienlayers at " << i << std::endl << gradientLayers.at(i) * (learnRate/miniBatch.size()) << std::endl;
        this->layers.at(i) = this->layers.at(i) - gradientLayers.at(i) * (learnRate/miniBatch.size());
        this->biases.at(i) = this->biases.at(i) - gradientBiases.at(i) * (learnRate/miniBatch.size());
    }
}

void NetworkFast::backProp(std::tuple<Matrix2,Matrix2> example,std::vector<Matrix2>& gradientLayers, std::vector<Matrix2>& gradientBiases)
{
    //opptater gradientLayers og Biases
    std::vector<Matrix2> deltaGradientLayers = makeEmptyLayers();
    std::vector<Matrix2> deltaGradientBiases = makeEmptyBiases();
    Matrix2 a1 = std::get<0>(example);
    std::vector<Matrix2> activations = {a1};
    Matrix2 emtyMatrix2;
    std::vector<Matrix2> z = {emtyMatrix2};
    Matrix2 outError; 
    //feedforward 
    for(int layer = 1; layer < numLayers; layer++){
        Matrix2 zl = layers.at(layer) * activations.at(layer-1) + biases.at(layer);
        z.push_back(zl);
        zl.applyActivationFunc("sigmoid");
        activations.push_back(zl);
    }
    //backProp
    Matrix2 dcOut = activations.at(numLayers-1) - std::get<1>(example);
    Matrix2 sigmoidPrime = z.at(numLayers-1);
    sigmoidPrime.sigmoidPrime(); //Mulighet for å bruke activations.at rett inn i sigPrime
    outError = (dcOut.hademart(sigmoidPrime));
    gradientBiases.at(numLayers-2) = gradientBiases.at(numLayers-2) + outError; 
    gradientLayers.at(numLayers-2) = gradientLayers.at(numLayers-2) + activations.at(numLayers - 2) * outError;
    for(int layer = numLayers-3; layer >= 0; layer--){
        Matrix2 sigmoidPrime = z.at(layer + 1);
        outError = (layers.at(layer + 1).transpose() * outError).hademart(sigmoidPrime);
        gradientBiases.at(layer) = gradientBiases.at(layer) + outError;
        gradientLayers.at(layer) = gradientLayers.at(layer) + activations.at(layer) * outError;
    } 
}

void NetworkFast::backProp2(std::tuple<Matrix2,Matrix2> example,std::vector<Matrix2>& gradientLayers, std::vector<Matrix2>& gradientBiases){
    //opptater gradientLayers og Biases
    std::vector<Matrix2> deltaGradientLayers = makeEmptyLayers();
    std::vector<Matrix2> deltaGradientBiases = makeEmptyBiases();
    Matrix2 a1 = std::get<0>(example);
    std::vector<Matrix2> activations = {a1};
    std::vector<Matrix2> z;  
    //feedforward 
    for (int layer = 1; layer < numLayers; ++layer) {
        Matrix2 zl = layers.at(layer - 1) * activations.at(layer - 1) + biases.at(layer - 1);
        z.push_back(zl);
        zl.applyActivationFunc("sigmoid");
        activations.push_back(zl);
    }
    //backProp
    Matrix2 dcOut = activations.back() - std::get<1>(example);
    Matrix2 sigmoidPrime = z.back();
    sigmoidPrime.sigmoidPrime();
    Matrix2 outError = dcOut.hademart(sigmoidPrime);
    gradientBiases.at(numLayers - 2) = gradientBiases.at(numLayers - 2) + outError;
    gradientLayers.at(numLayers - 2) = gradientLayers.at(numLayers - 2) + (outError * activations.at(numLayers - 2).transpose());
    for (int layer = numLayers - 3; layer >= 0; --layer) {
        Matrix2 sp = z.at(layer);
        sp.sigmoidPrime();
        outError = (layers.at(layer + 1).transpose() * outError).hademart(sp);
        gradientBiases.at(layer) = gradientBiases.at(layer) + outError;
        gradientLayers.at(layer) = gradientLayers.at(layer) + outError * activations.at(layer).transpose();
    }
}

std::vector<Matrix2> NetworkFast::makeEmptyLayers(){
    std::vector<Matrix2> zeroLayers;
    for(int i = 1; i < numLayers; i++){
        Matrix2 addLayer(neurovec.at(i), neurovec.at(i-1));
        addLayer.setValue(0);
        zeroLayers.push_back(addLayer);
    }
    return zeroLayers;
}
std::vector<Matrix2> NetworkFast::makeEmptyBiases(){
    std::vector<Matrix2> zeroBiases;
    for(int i = 1; i < numLayers; i++){
        Matrix2 addBias(neurovec.at(i),1);
        addBias.setValue(0);
        zeroBiases.push_back(addBias);
    }
    return zeroBiases;
}


void NetworkFast::evaluate(std::vector<std::tuple<Matrix2,Matrix2>> testData){
    int numRight = 0;
    for(std::tuple<Matrix2,Matrix2> test : testData){
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
    std::cout << " " << (static_cast<double>(numRight) / testData.size()) * 100.0 << " prosent riktige" << std::endl;
}

void NetworkFast::saveNetworkToFile(const std::string& filename){
    std::ofstream outFile(filename);
    // lagre std::vector<int> neurovec;
    outFile << "neurovec:" << std::endl;
    outFile << neurovec;
    outFile << 'n' << std::endl;
    outFile << layers;
    outFile << biases;
    //lagre std::vector<Matrix2> layers og std::vector<Matrix2> biases;
}

void NetworkFast::printNetworkToTerminal(){
    std::cout << "neurovec: \n" << neurovec << "\n Wheights: \n" << layers << "\n Biases: \n" << biases;
    //lagre std::vector<Matrix2> layers og std::vector<Matrix2> biases;
}

