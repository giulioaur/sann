/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  LOGGER CLASS HEADER                                *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/
#ifndef S_UTILITY_LOGGER_S
#define S_UTILITY_LOGGER_S

#include <string>
#include <fstream>
#include <mutex>

namespace sann{
namespace utility{

/// This class takes care of create and update a log files. This class is thread-safe.
class Logger{

    static std::ofstream logFile;
    static std::mutex mtx;
    static int initSession;

public: 

    enum class type{NONE, INFO, WARN, ERROR};

    static void writeLog(const std::string &text, Logger::type type = Logger::type::INFO, bool showDate = true);
};

}
}

#endif