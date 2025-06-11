#include "xor.h"
#include "dataPath.h"

void testXor(std::string name){
    if(name == "network"){
        std::vector<int> inputVector = {2,4,2};
        Matrix inputLayer(2,1);
        Network net(inputVector,inputLayer);

         //Make traing data 
         std::vector<std::tuple<Matrix, Matrix>> xorData;

         for (int i = 0; i < 2; i++) {
             for(int j = 0; j < 2; j++){
                 std::vector<std::vector<double>> localVec = {{static_cast<double> (i)},{static_cast<double>(j)}};
                 std::vector<std::vector<double>> outVec = {{1},{0}};
                 Matrix input(localVec);
                 if (i xor j){
                     outVec = {{0},{1}};
                 }
                 Matrix output(outVec);
                 xorData.push_back({input, output});
             }
         }
 

        net.applySGD(xorData, 10000, 4, 1.0, xorData);
    }
    else if(name == "networkFast"){
        //Initelaize network 
        std::vector<int> inputVector = {2,4,2};
        Matrix2 inputLayer(2,1);
        NetworkFast net(inputVector,inputLayer);

        //Make traing data 
        std::vector<std::tuple<Matrix2, Matrix2>> xorData;

        for (int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++){
                std::vector<std::vector<double>> localVec = {{static_cast<double> (i)},{static_cast<double>(j)}};
                std::vector<std::vector<double>> outVec = {{1},{0}};
                Matrix2 input(localVec);
                if (i xor j){
                    outVec = {{0},{1}};
                }
                Matrix2 output(outVec);
                xorData.push_back({input, output});
            }
        }

        //Do cool stufffffff
        net.applySGD(xorData, 1000, 4, 1.0, xorData);
        std::string filePath = dataPath +"XorToFile.txt";
        net.saveNetworkToFile(filePath);
        NetworkFast net2(filePath);
        net2.printNetworkToTerminal();
    }
    else{
        std::cout << "Du må gi inn navnet på hvilket netverk du vil bruke (network) eller (networkFast) " << std::endl;
    }
}