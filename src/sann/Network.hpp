/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  NETWORK CLASS HEADER                               *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/

#ifndef S_NETWORK_S
#define S_NETWORK_S

// System libaries include.
#include <vector>

// My include
#include "Layer.hpp"
#include "Estimator.hpp"
#include "math/Func.hpp"
#include "math/Plotter.hpp"

namespace sann{

/// This is the core class, that represents the whole Neural Network.
class Network{
private:
    // ATTRIBUTES

    std::vector<Layer> layers;
    size_t inputSize;
    std::shared_ptr<std::function<std::vector<double>(const std::vector<double> &target, 
        const std::vector<double> &out)>> errorFunc;

    // METHODS
    
    void trainStep(const std::vector<double> &trainPattern, const std::vector<double> &expectedResults, 
                                    sann::Estimator &est);

public:
    // STATIC ATTRIBUTES
    static sann::Estimator &nullEstimator;
    typedef std::function<std::vector<double>(const std::vector<double> &target, 
        const std::vector<double> &out)> error_func;

    // CONSTRUCTORS

    Network();
    Network(const std::vector<size_t> &layers, const math::Func &activationFunc, const Layer::weights_initializer &init);
    Network(const std::vector<size_t> &layers, const std::vector<math::Func> &activationFuncs,
                const Layer::weights_initializer &init);
    Network(const Network &net);

    // OPERATORS

    Network& operator = (const Network &rhs);
    Network& operator = (Network &&rhs);

    // METHODS

    void setParameters(const sann::parameters &hyperP);
    void setWeights(const std::vector<std::vector<double>> &weights);
    void setWeights(const std::vector<weightsMatrix> &weights);
    void setWeights(std::vector<weightsMatrix> &&weights);
    void setRandomWeights();
    void setErrorFunction(const error_func &error);
    std::vector<weightsMatrix> getWeights() const;
    std::vector<std::size_t> getlayersSizes() const;

    // Computation
    std::vector<double> compute(const std::vector<double> &inputs);

    // Train.
    void train(const sann::dataSet &trainingSet, sann::Estimator &est, const sann::parameters &hyperPar);
    void train(const sann::dataSet &trainingSet, const sann::dataSet &testSet, sann::Estimator &trainEst,
                sann::Estimator &testEst, const sann::parameters &hyperPar);
};

}



#endif