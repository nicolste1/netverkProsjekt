#include <xor.h>

void testXor(std::string name){
    if(name == "network"){
        std::vector<int> inputVector = {2,4,2};
        Matrix inputLayer(2,1);
        Network net(inputVector,inputLayer);
        std::vector<std::tuple<Matrix, Matrix>> xorData;

        std::vector<std::vector<std::vector<double>>> inputs = {
            {{0}, {0}},
            {{0}, {1}},
            {{1}, {0}},
            {{1}, {1}}
        };

        std::vector<int> expected = {0, 1, 1, 0}; // XOR-resultater

        for (int i = 0; i < inputs.size(); ++i) {
            Matrix input(inputs[i]); // 2x1 input-matrise

            std::vector<std::vector<double>> outputVec = {
                { expected[i] == 0 ? 1.0 : 0.0 },
                { expected[i] == 1 ? 1.0 : 0.0 }
            };
            Matrix output(outputVec); // 2x1 output-matrise

            xorData.push_back({input, output});
        }
        net.applySGD(xorData, 10000, 4, 1.0, xorData);
    }
    else if(name == "networkFast"){
        std::vector<int> inputVector = {2,4,2};
        Matrix2 inputLayer(2,1);
        NetworkFast net(inputVector,inputLayer);
        std::vector<std::tuple<Matrix2, Matrix2>> xorData;
        std::vector<std::vector<std::vector<double>>> inputs = {
            {{0}, {0}},
            {{0}, {1}},
            {{1}, {0}},
            {{1}, {1}}
        };

        std::vector<int> expected = {0, 1, 1, 0}; // XOR-resultater

        for (int i = 0; i < inputs.size(); ++i) {
            Matrix2 input(inputs[i]); // 2x1 input-matrise
            std::vector<std::vector<double>> outputVec = {
                { expected[i] == 0 ? 1.0 : 0.0 },
                { expected[i] == 1 ? 1.0 : 0.0 }
            };
            Matrix2 output(outputVec); // 2x1 output-matrise
            xorData.push_back({input, output});
        }
        //net.applySGD(xorData, 1000, 4, 1.0, xorData);
        net.saveNetworkToFile("XorToFile.txt");
        NetworkFast net2("XorToFile.txt");
        net2.printNetworkToTerminal();
    }
    else{
        std::cout << "Du må gi inn navnet på hvilket netverk du vil bruke (network) eller (networkFast) " << std::endl;
    }
}