/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  RANDOMIZER CLASS HEADER                            *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/

#ifndef S_MATH_RANDOM_S
#define S_MATH_RANDOM_S

// System libraries include.
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <iostream>
#include <random>

namespace sann{
namespace math{

/// This class offers only static methods for random calculations.

class Randomizer{
private:
    static int initSeed;
public:
    // METHOD

    /**
     * @brief Compute a random number between the given bounds.
     *        NB: this function is not thread safe, use randomRange with seed parameter instead.
     * 
     * @tparam T - The type of the element of the vector.
     * @param min - The lower bound.
     * @param max - The upper bound.
     * @return template<typename T> T - The random number.
     */
    template <typename T> 
    static T randomRange(const T min, const T max){
        double value = (double)rand() / RAND_MAX;
        return min + value * (max - min);
    }

    /**
     * @brief Compute a random number between the given bounds.
     * 
     * @tparam T - The type of the element of the vector.
     * @param min - The lower bound
     * @param max - The upper bound.
     * @param seed - The state-keeping.
     * @return template<typename T> T - The random number.
     */
    template <typename T> 
    static T randomRange(const T min, const T max, unsigned int &seed){
        double value = (double)rand_r(&seed) / RAND_MAX;
        return min + value * (max - min);
    }

    /**
     * @brief Compute a vector of random number between the given bounds.
     * 
     * @tparam T - The type of the element of the vector.
     * @param min - The lower bound
     * @param max - The upper bound.
     * @param size - The size of the vector. 
     * @return template<typename T> T - The vector of random number.
     */
    template <typename T>
    static std::vector<T> randomRangeVector(T min, T max, size_t size){
        std::vector<T> randomVector;
        randomVector.reserve(size);

        for(size_t i = 0; i < size; i++){
            double value = (double)rand() / RAND_MAX;
            randomVector.push_back(min + value * (max - min));
        }

        return randomVector;
    }

    /**
     * @brief Compute a vector of random number between the given bounds.
     * 
     * @tparam T - The type of the element of the vector.
     * @param min - The lower bound
     * @param max - The upper bound.
     * @param size - The size of the vector. 
     * @param seed - The state-keeping.
     * @return template<typename T> T - The vector of random number.
     */
    template <typename T>
    static std::vector<T> randomRangeVector(const T &min, const T &max, size_t size, unsigned int &seed){
        std::vector<T> randomVector;
        randomVector.reserve(size);

        for(size_t i = 0; i < size; i++){
            double value = (double)rand_r(&seed) / RAND_MAX;
            randomVector.push_back(min + value * (max - min));
        }

        return randomVector;
    }

    /**
     * @brief Returns a vector of random number taken from a Gaussian distribution.
     * 
     * @tparam T - The type of the element of the vector.
     * @param mean - The mean of the Gaussian distribution.
     * @param stddev - The standard deviation.
     * @param size - The size of the vector.
     * @return std::vector<T> - The vector of random numbers.
     */
    template <typename T>
    static std::vector<T> randomGaussianVector(const T &mean, const T &stddev, size_t size){
        std::vector<T> randomVector;
        std::normal_distribution<double> gaussian(mean, stddev);
        std::random_device rd; 
        std::mt19937 gen(rd());
        // std::default_random_engine gen;
        randomVector.reserve(size);

        for(size_t i = 0; i < size; i++)
            randomVector.push_back(gaussian(gen));

        return randomVector;
    }
};

}
}


#endif