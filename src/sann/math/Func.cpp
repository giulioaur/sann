/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  FUNC CLASS FILE                                    *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/

#include "Func.hpp"

namespace sann{
namespace math{
    /**
     * @brief The default constructor.
     * 
     * @param func - The function.
     * @param derivative - The derivative of the function.
     */
    Func::Func(const std::function<double(const double)> &func, const std::function<double(const double)> &derivative) :
         func(func), deriv(derivative){}

    Func::Func(const Func &func): func(func.func), deriv(func.deriv){}
    
    /**
     * @brief Computes the function.
     * 
     * @param input - The input of the function.
     * @return double - The computed value.
     */
    double Func::call(const double input) const{
        return this->func(input);
    }

    /**
     * @brief Computes the derivative of the function.
     * 
     * @param input - The input of the derivative.
     * @return double - The computed derivative.
     */
    double Func::derivative(const double input) const{
        return this->deriv(input);
    }


    // STATIC FUNCTION

    /**
     * @brief Returns the linear function.
     * 
     * @return Func - The linear function.
     */
    Func Func::linear = Func(
        [](const double x) -> double{
            return x;
        },
        [](const double x) -> double{
            return 1;
        }
    );

    /**
     * @brief Returns the sigmoid function with a slope parameters.
     * 
     * @param slope - The slope parameters (1 by dafault).
     * @return Func - The sigmoid function.
     */
    Func Func::sigmoid = Func(
        [](const double x) -> double{ // 1 / (1 + exp(-ax))
            return 1.0 / ( 1.0 + exp( -x ) );
        },
        [](const double x) -> double{ // fs(x) (1 - fs(x))
            double ex = exp(x);
            return ex / pow(ex + 1, 2);
        }
    );

    /**
     * @brief Returns the tanh function.
     * 
     * @return Func - The tanh function.
     */
    Func Func::tanH = Func(
        [](const double x) -> double{ // (2 / (1 + e^(-2x))) - 1
            return (2 / (1 + exp(-x * 2))) - 1;
        },
        [](const double x) -> double{ // 1 - ftanh(x)^2
            return 1 - pow((2 / (1 + exp(-x * 2))) - 1, 2);
        }
    );

    /**
     * @brief Returns the relu function.
     * 
     * @return Func - The relu function.
     */
    Func Func::ReLU = Func(
        [](const double x) -> double{ // max(x, 0)
            return x > 0 ? x : 0;
        },
        [](const double x) -> double{ // x > 0 ? 1 : 0;
            return x > 0 ? 1 : 0;
        }
    );
}
}