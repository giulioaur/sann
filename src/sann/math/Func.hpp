/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  FUNC CLASS HEADER                                  *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/

#ifndef S_MATH_FUNC_S
#define S_MATH_FUNC_S

// System libraries include.
#include <vector>
#include <math.h>
#include <functional>

namespace sann{

namespace math{

/// This class represent a mathematical function used as activation function.
class Func{

private:

    // ATTRIBUTE

    std::function<double(const double)> func; // The function.
    std::function<double(const double)> deriv; // The function's derivative.

public:
    
    // Default constructor.
    Func(const std::function<double(const double)> &func, const std::function<double(const double)> &derivative);
    // Copy constructor.
    Func(const Func &func);

    // METHODS

    double call(const double input) const;
    double derivative(const double input) const;

    // STANDARD FUNCTION

    static Func linear;
    static Func sigmoid;
    static Func tanH;
    static Func ReLU;
};

}

}

#endif