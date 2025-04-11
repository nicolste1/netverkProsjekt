#include <visuelMnist.h>
#include <mnist_loader.hpp>
#include <windows.h>  //For ventetid mellom hvert bilde
#include <widgets\Button.h>
#include <random>

void testMNISTvindu() {
}

void testDrawVindu() {

    //Dimensjoner på vindu
    const int PixelwidthDraw = 20;
    const int PixelheightDraw = 20;
    const int sideBarWidth = 10 * PixelwidthDraw;
    const int sideBarPadding = PixelwidthDraw;
    const int heightPadding = 28 * PixelheightDraw / 14;
    const int titleBarHeight = 4 * PixelwidthDraw;
    const int fontTitle = (titleBarHeight / 1.25);
    const int fontGuess = PixelwidthDraw * 2.5;
    const int width = 28 * PixelwidthDraw + sideBarWidth;
    const int height = 28 * PixelheightDraw + titleBarHeight;
    TDT4102::AnimationWindow win{100, 100, width, height, "Test Vindu"};
    

    //Laste netverk fra fil
    NetworkFast net("VisuelMNISTfile.txt");

    //TestData fra MNIST
    int antPicFromMNIST = 1000;
    std::vector<std::tuple<std::vector<double>, double>> testData = returnTestVector(antPicFromMNIST);

    //Vector som lagrer tegning fra bruker
    std::vector<double> drawVec(28 * 28, 0.0);
    

    //Buttons 
    const int buttonWidth = sideBarWidth - 2 * sideBarPadding;
    const int buttonHeight = 2 * heightPadding;

    TDT4102::Point guessPoint = {28 * PixelwidthDraw + sideBarPadding, titleBarHeight + 2 * heightPadding};
    TDT4102::Point guessTextPoint = {28 * PixelwidthDraw, titleBarHeight + 5 * heightPadding};
    TDT4102::Button guessButton {guessPoint, buttonWidth, buttonHeight, "Guess"};
    bool didAguess = false;
    int netPredict;
    guessButton.setCallback([&drawVec, &didAguess, &netPredict, &net](){
        netPredict = net.feedforward(drawVec);
        didAguess = true;
        std::fill(drawVec.begin(), drawVec.end(), 0.0);
    });
    win.add(guessButton);

    //MNIST data button
    TDT4102::Point MNISTpoint = {28 * PixelwidthDraw + sideBarPadding, titleBarHeight + 8 * heightPadding};
    TDT4102::Button MNISTbutton {MNISTpoint, buttonWidth, buttonHeight, "MNIST"};
    MNISTbutton.setCallback([&drawVec, &didAguess, &netPredict, &net, &testData, &antPicFromMNIST](){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, antPicFromMNIST);
        int randomIndexMNIST = dist(gen);

        drawVec = std::get<0> (testData.at(randomIndexMNIST));
        netPredict = net.feedforward(drawVec);
        didAguess = true;
    });
    win.add(MNISTbutton);
    
    
    /* Mulig knapp for å legge til treningsdata som man kan trene på 
    TDT4102::Point trainDataPoint = {28 * PixelwidthDraw + sideBarPadding, titleBarHeight + 8 * heightPadding};
    TDT4102::Button trainDataButton {trainDataPoint, buttonWidth, buttonHeight, "addData"};
    win.add(trainDataButton);
    */

   //Åpne vindu for tegning
    while (!win.should_close()) {
        
        //SideBar
        win.draw_line(TDT4102::Point{28 * PixelwidthDraw - 1, 0}, TDT4102::Point{28 * PixelwidthDraw - 1 , height - 1}, TDT4102::Color::red);
        win.draw_text(TDT4102::Point{28 * PixelwidthDraw, 1 }, "Sidebar", TDT4102::Color::black,  fontTitle, TDT4102::Font::arial);
    
        //Title
        win.draw_text(TDT4102::Point{1, 1}, "Draw here", TDT4102::Color::black,  fontTitle, TDT4102::Font::arial);
        win.draw_line(TDT4102::Point{0 , titleBarHeight}, TDT4102::Point{width - 1, titleBarHeight}, TDT4102::Color::red);

        //Skriv opp Guess
        if(didAguess){
            win.draw_text(guessTextPoint,"Guess: " + std::to_string(netPredict), TDT4102::Color::black,  fontGuess, TDT4102::Font::arial);
        }

        //UpdateDrawVec
        if(win.is_left_mouse_button_down()){
            TDT4102::Point point = win.get_mouse_coordinates();
            if(point.x > 0 && point.x < 26 * PixelwidthDraw && point.y > titleBarHeight + 2 * PixelheightDraw && point.y < (height - 1 - PixelheightDraw)){
                int iPos = static_cast<int> ((point.y - titleBarHeight) / PixelheightDraw);
                int jPos = static_cast<int> (point.x / PixelwidthDraw);
                drawVec.at(iPos*28 + jPos) = 1;
                drawVec.at(iPos*28 + (jPos + 1)) = 1;
                drawVec.at((iPos*28) + (jPos + 2)) = 1;

                drawVec.at((iPos + 1)*28 + jPos) = 1;
                drawVec.at((iPos + 1)*28 + (jPos + 1)) = 1;              


            }
        }


        //Draw 
        for(int i = 0; i < 28; i++){
            for(int j = 0; j < 28; j++){
                if (drawVec.at(i*28 + j) != 0){
                    win.draw_rectangle(TDT4102::Point{PixelwidthDraw * j,PixelheightDraw * i + titleBarHeight}, PixelwidthDraw, PixelheightDraw, TDT4102::Color::black);
                }
            }
        }


        win.next_frame();    
    }
}

