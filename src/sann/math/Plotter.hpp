/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  PLOTTER CLASS HEADER                               *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/

#ifndef S_UTILITY_PLOTTER_S
#define S_UTILITY_PLOTTER_S

// System libraries includes.
#include <vector>
#include <string>
#include <functional>
#include <initializer_list>

namespace sann{     
namespace math {
/// This class allows to plot functions and data on csv file. 
class Plotter{
    std::string plotName;

public:

    Plotter(const std::string &plotName, const bool clean = true);

    // METHODS

    void plotFunction(const std::initializer_list<std::vector<double>> &list) const;
    void plotFunction(const std::vector<double> &x, const std::vector<double> &y) const;
    void plotFunction(const std::vector<double> &x, std::function<double(double)> fnc) const;
    void plotPoints(const std::vector<double> &x, const std::vector<double> &y, const std::vector<short> &classes = {}) const;
};

}
}

#endif