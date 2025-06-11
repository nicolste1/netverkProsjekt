#include "dataPath.h"


std::string& fileExist(std::string& filePath) { 
    //Check if filepaths are valid
    bool completeLoad = false;
    while (!completeLoad) {
        try {
            if (!std::filesystem::exists(std::filesystem::path(filePath))) {
                throw std::runtime_error("Could not open file from filepath: " + filePath);
            }
            std::cout << filePath + " is an valid filePath";
            completeLoad = true;
        }
        catch (const std::exception& e) {
            std::string ans;
            std::cout << e.what() << "\nDo you have new file path? [Y/N]: " << std::endl;
            std::cin >> ans;
            if (ans == "Y" || ans == "y") {
                std::cout << "Enter filePath, working directory are: " << std::filesystem::current_path() << std::endl;
                std::cin >> filePath;
            }

            else {
                std::cout << "Terminating program" << std::endl;
                std::exit(1);
            }
        }
    }
    return filePath;
}