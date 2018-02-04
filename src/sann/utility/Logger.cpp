/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  LOGGER CLASS FILE                                  *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/
#include "Logger.hpp"

// Other system includes.
#include <ctime>

using namespace std;

const string typeName[4] = {"", "INFO", "WARNING", "ERROR"};

namespace sann{
namespace utility{

    ofstream Logger::logFile = ofstream{"./info.log", std::ios::out | std::ios::app};
    mutex Logger::mtx;
    // Write on the file the start point of new session.
    int Logger::initSession = []() -> int { 
        Logger::logFile << "\n*****************************NEW SESSION*****************************" << endl; 
        return 1; 
    }();

    /**
     * @brief Writes on the log file.
     * 
     * @param text - What to write.
     * @param type - The type of the information to write.
     * @param showDate - True if the date must be shown, false otherwise.
     */
    void Logger::writeLog(const string &text, Logger::type type, bool showDate){
        // Write in the log file.
        Logger::mtx.lock();

        if(showDate){
            time_t result = time(nullptr);
            string currentTime = asctime(localtime(&result));  // Human readable time.
            currentTime.pop_back(); // Delete newline.
            Logger::logFile << currentTime << "\t";
        }
        if(type != Logger::type::NONE)  Logger::logFile <<  typeName[(short)type] << '\t';
        Logger::logFile << text << endl; 

        Logger::mtx.unlock();
    }

}
}