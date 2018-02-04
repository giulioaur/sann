/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  LAYER CLASS HEADER                                 *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/

#ifndef S_LAYER_S
#define S_LAYER_S

// System libraries include.
#include <vector>
#include <memory>

// My includes.
#include "dataStructures.h"
#include "math/Func.hpp"

namespace sann{

/// This class represents a layer of the neural network.
class Layer{
private:
    // ATTRIBUTES

    short level;
    size_t neurons;
    weightsMatrix weights, currErrors, prevErrors;
    std::vector<double> lastNet;
    math::Func func;
    
    // METHODS

    std::vector<double> computeNets(std::vector<double> inputs) const;
public:
    // TYPEDEF

    typedef std::function<weightsMatrix(const std::size_t m, const std::size_t n)> weights_initializer;

    // CONSTRUCTORS

    Layer();
    Layer(std::size_t numOfNeurons, const math::Func &activationFunc, const short level = -1);
    Layer(const Layer &lay);

    // METHODS
    
    void setWeights(const std::vector<double> &weights);
    void setWeights(const weightsMatrix &weights);
    void setWeights(weightsMatrix &&weights);
    void setWeights(const weights_initializer &init, const size_t n);
    const weightsMatrix& getWeights() const;
    size_t getSize() const;

    // COMPUTATION

    std::vector<double> feed_forward(const std::vector<double> &inputs);
    std::vector<double> back_propagation(const std::vector<double> &inputs, const std::vector<double> &errors);
    void updateWeights(const sann::parameters &hyperP);
};

}

#endif