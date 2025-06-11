#include "visuelMnist.h"
#include "mnist_loader.hpp"
#include "dataPath.h"
#include <thread>

#include <windows.h>  //For ventetid mellom hvert bilde
#include <widgets\Button.h>
#include <random>


//Dimensjoner på vindu
const int PixelwidthDraw = 20;
const int PixelheightDraw = 20;
const int sideBarWidth = 10 * PixelwidthDraw;
const int sideBarPadding = PixelwidthDraw;
const int heightPadding = MNISTsize * PixelheightDraw / 14;
const int titleBarHeight = 4 * PixelwidthDraw;
const int fontTitle = (titleBarHeight / 1.25);
const int fontGuess = PixelwidthDraw * 2.5;
const int width = MNISTsize * PixelwidthDraw + sideBarWidth;
const int height = MNISTsize * PixelheightDraw + titleBarHeight;
const int optimalDrawBoxSize = PixelwidthDraw * 4;



void testDrawWindow() {

    
    TDT4102::AnimationWindow win{100, 100, width, height, "Test Vindu"};

    //Laste netverk fra fil
    std::string pathToNetwork = {dataPath +"VisuelMNISTfile.txt"};
    NetworkFast net(pathToNetwork);

    //TestData fra MNIST
    int antPicFromMNIST = 1000;
    std::vector<std::tuple<std::vector<double>, double>> testData = returnTestVector(antPicFromMNIST);

    //Lagredata
    std::vector<std::tuple<std::vector<double>, double>> PersonalTrainData;
    int numPersonalDuplicate = 10;
    int numAugment = 10; 
    
    //Vector som lagrer tegning fra bruker
    std::vector<double> drawVec(MNISTsize * MNISTsize, 0.0);
    

    //Buttons_______________________________________________________________________________________-
    const unsigned int buttonWidth = sideBarWidth - 2 * sideBarPadding;
    const unsigned int buttonHeight = 2 * heightPadding;
    bool wantToTrain = false;
    bool didAguess = false;
    int netPredict;

    //Guess button
    TDT4102::Point guessPoint = {MNISTsize * PixelwidthDraw + sideBarPadding, titleBarHeight + 2 * heightPadding};
    TDT4102::Point guessTextPoint = {MNISTsize * PixelwidthDraw, titleBarHeight + 5 * heightPadding};
    TDT4102::Button guessButton {guessPoint, buttonWidth, buttonHeight, "Guess"};
    guessButton.setCallback([&drawVec, &didAguess, &netPredict, &net, &wantToTrain](){
        netPredict = net.feedforward(drawVec);
        didAguess = true;
        wantToTrain = false;
        std::fill(drawVec.begin(), drawVec.end(), 0.0);
    });
    win.add(guessButton);

    //MNIST data button
    TDT4102::Point MNISTpoint = {MNISTsize * PixelwidthDraw + sideBarPadding, titleBarHeight + 8 * heightPadding};
    TDT4102::Button MNISTbutton {MNISTpoint, buttonWidth, buttonHeight, "MNIST"};
    MNISTbutton.setCallback([&drawVec, &didAguess, &netPredict, &net, &testData, &antPicFromMNIST, &wantToTrain](){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, antPicFromMNIST);
        int randomIndexMNIST = dist(gen);

        drawVec = std::get<0> (testData.at(randomIndexMNIST));
        netPredict = net.feedforward(drawVec);

        didAguess = true;
        wantToTrain = false;
    });
    win.add(MNISTbutton);
    
    
    //Mulig knapp for å legge til treningsdata som man kan trene på 
    TDT4102::Point trainDataInfoPoint = {MNISTsize * PixelwidthDraw + sideBarPadding, titleBarHeight + 10 * heightPadding};
    TDT4102::Point trainDataInfoPoint2 = {MNISTsize * PixelwidthDraw + sideBarPadding, titleBarHeight + 11 * heightPadding};

    TDT4102::Point trainDataPoint = {MNISTsize * PixelwidthDraw + sideBarPadding, titleBarHeight + 12 * heightPadding};
    TDT4102::Button trainDataButton {trainDataPoint, buttonWidth, buttonHeight, "Train"};
    trainDataButton.setCallback([&drawVec, &didAguess, &netPredict, &net, &wantToTrain](){
        didAguess = false;
        wantToTrain = true;
        netPredict = 1;
        std::fill(drawVec.begin(), drawVec.end(), 0.0);
    });
    win.add(trainDataButton);

    bool spaceWasDown = false;
    bool eWasDown = false;
    bool trainInProgress = false;
    std::mutex netMutex;
    

    

    //Open window for drawing___________________________________________________________________________________
    while (!win.should_close()) {
        
        //SideBar
        win.draw_line(TDT4102::Point{MNISTsize * PixelwidthDraw - 1, 0}, TDT4102::Point{MNISTsize * PixelwidthDraw - 1 , height - 1}, TDT4102::Color::red);
        win.draw_text(TDT4102::Point{MNISTsize * PixelwidthDraw, 1 }, "Sidebar", TDT4102::Color::black,  fontTitle, TDT4102::Font::arial);
    
        //Title
        win.draw_text(TDT4102::Point{1, 1}, "Draw here", TDT4102::Color::black,  fontTitle, TDT4102::Font::arial);
        win.draw_line(TDT4102::Point{0 , titleBarHeight}, TDT4102::Point{width - 1, titleBarHeight}, TDT4102::Color::red);

        //optimal draw box
        win.draw_line(TDT4102::Point{optimalDrawBoxSize , titleBarHeight + optimalDrawBoxSize}, TDT4102::Point{MNISTsize * PixelwidthDraw - 1 - optimalDrawBoxSize, titleBarHeight + optimalDrawBoxSize}, TDT4102::Color::blue);
        win.draw_line(TDT4102::Point{MNISTsize * PixelwidthDraw - 1 - optimalDrawBoxSize, titleBarHeight + optimalDrawBoxSize}, TDT4102::Point{MNISTsize * PixelwidthDraw - 1 - optimalDrawBoxSize, height - optimalDrawBoxSize}, TDT4102::Color::blue);
        win.draw_line(TDT4102::Point{MNISTsize * PixelwidthDraw - 1 - optimalDrawBoxSize, height - optimalDrawBoxSize}, TDT4102::Point{optimalDrawBoxSize , height - optimalDrawBoxSize}, TDT4102::Color::blue);
        win.draw_line(TDT4102::Point{optimalDrawBoxSize , height - optimalDrawBoxSize}, TDT4102::Point{optimalDrawBoxSize , titleBarHeight + optimalDrawBoxSize}, TDT4102::Color::blue);
        //Write Guess/predicted
        if(didAguess || wantToTrain){
            std::string guessOrTrainText = "Guess: ";
            if(wantToTrain){
                guessOrTrainText = "Write: ";
                win.draw_text(trainDataInfoPoint, "Draw next: SPACE", TDT4102::Color::black,  fontGuess / 3, TDT4102::Font::arial);
                win.draw_text(trainDataInfoPoint2, "Update net: E", TDT4102::Color::black,  fontGuess / 3, TDT4102::Font::arial);
            }            
            win.draw_text(guessTextPoint,guessOrTrainText + std::to_string(netPredict), TDT4102::Color::black,  fontGuess, TDT4102::Font::arial);
        }

        //Lagre treningsbilder 
        bool spaceIsDown = win.is_key_down(KeyboardKey::SPACE);
        bool eIsDown = win.is_key_down(KeyboardKey::E);

        if(wantToTrain){
            if(spaceIsDown && !spaceWasDown){
                PersonalTrainData.push_back({drawVec,netPredict});                
                std::fill(drawVec.begin(), drawVec.end(), 0.0);
                netPredict = (netPredict + 1) % 10; //Når vi trener går vi fra 0-9 
            }


            //optatere netverk i egen thread som kjører simultant
            if(eIsDown && !eWasDown && PersonalTrainData.size() != 0 && !trainInProgress){
                auto localPersonalTrainData = PersonalTrainData;
                std::thread trainThread([= , &net, &netMutex, &trainInProgress]() mutable {
                    {
                        std::lock_guard<std::mutex> lock(netMutex);
                        trainInProgress = true;
                    }

                    //augmentering 
                    std::cout << "Network update in progress: " << std::endl;
                    std::vector<std::tuple<std::vector<double>, double>> localContainer;
                    for(auto & v : localPersonalTrainData){ 

                        std::vector<double> augmentVec = std::get<0> (v);

                        while(checkIfColumClear(augmentVec , MNISTsize-1))//kolonnen helt til høyre ikke er fylt så flytter vi en til høyre. 
                        {
                            //skyv vector en til venstre eller høyre som tilsvarer 10 piksler med nåværende konfig.  
                            augmentVec = moveLeftRight(augmentVec, "R");
                            localContainer.emplace_back(augmentVec, netPredict);
                        }
                        
                        augmentVec = std::get<0> (v);

                        while(checkIfColumClear(augmentVec, 0)){
                            augmentVec = moveLeftRight(augmentVec, "L");
                            localContainer.emplace_back(augmentVec,netPredict);
                        }

                    }
                    localPersonalTrainData.insert(localPersonalTrainData.end(), localContainer.begin(), localContainer.end());
                    std::cout << "augemntering done! \n";
                    //Oversampling 
                    oversample(localPersonalTrainData, numPersonalDuplicate);

                    std::cout << "oversample done! \n";
                    NetworkFast netCopy;
                    {
                        std::lock_guard<std::mutex> lock(netMutex);
                        netCopy = net;
                    }

                    updatePersonalNet(netCopy, localPersonalTrainData);
                    {
                        std::lock_guard<std::mutex> lock(netMutex);
                        net = netCopy; 
                    }

                    std::cout << "network update done! \n";

                    {
                        std::lock_guard<std::mutex> lock(netMutex);
                         trainInProgress = false;
                    }
                });
            
                trainThread.detach();
            
                
            }
        }

        spaceWasDown = spaceIsDown;
        eWasDown = eIsDown;


        //UpdateDrawVec
        if(win.is_left_mouse_button_down()){
            TDT4102::Point point = win.get_mouse_coordinates();
            if(point.x > 0 && point.x < 26 * PixelwidthDraw && point.y > titleBarHeight + 2 * PixelheightDraw && point.y < (height - 1 - PixelheightDraw)){
                int iPos = static_cast<int> ((point.y - titleBarHeight) / PixelheightDraw);
                int jPos = static_cast<int> (point.x / PixelwidthDraw);
                drawVec.at(iPos*MNISTsize + jPos) = 1;
                drawVec.at(iPos*MNISTsize + (jPos + 1)) = 1;
                drawVec.at((iPos*MNISTsize) + (jPos + 2)) = 1;

                drawVec.at((iPos + 1)*MNISTsize + jPos) = 1;
                drawVec.at((iPos + 1)*MNISTsize + (jPos + 1)) = 1;              


            }
        }


        //Draw 
        for(int i = 0; i < MNISTsize; i++){
            for(int j = 0; j < MNISTsize; j++){
                if (drawVec.at(i*MNISTsize + j) != 0){
                    win.draw_rectangle(TDT4102::Point{PixelwidthDraw * j,PixelheightDraw * i + titleBarHeight}, PixelwidthDraw, PixelheightDraw, TDT4102::Color::black);
                }
            }
        }

        win.next_frame();    
    }
}

