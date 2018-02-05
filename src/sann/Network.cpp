/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  NETWORK CLASS FILE                                 *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/
#include "Network.hpp"

// Other system libraries include.
#include <stdexcept>
#include <cmath>

// Debug libraries.
#ifdef S_DEBUG_MODE_S
#include <iostream>
#include "math/Plotter.hpp"

using namespace sann::math;
#endif

using namespace std;

namespace sann{
    // Static variables definition.

    /**
     * @brief An estimator that does nothing. It can be used when the user wants no monitoring nor adding more
     *        stopping criteria to the training.
     * 
     */
    class NullEst : public Estimator{
        inline void init(std::size_t epoch){ }

        inline bool stoppingCriteria(){ return false; }

        inline void update(const std::vector<double> &out, const std::vector<double> &expected){ }

        inline void plot(){ }

        inline void terminate(){ }
    } realNullEstimator{};
    Estimator &Network::nullEstimator = realNullEstimator;

    auto mse = [](const vector<double> &targets, const vector<double> &results){
        vector<double> errors (targets.size());
        for(size_t i = 0; i < targets.size(); ++i){
            errors[i] = targets[i] - results[i];
        }
        return errors;
    };

    // CONSTRUCTORS

    /**
     * @brief Creates a new empty network.
     * 
     */
    Network::Network() : inputSize(0){ }

    /**
     * @brief Creates a fully connected network with random weights. All the neurons have the
     *        same activation function. The input size is the size of the first layer, the output
     *        size is the one of the last layer.
     *
     * @param layers - The vector with the size of each layer.
     * @param activationFunc - The activation function of the neurons.
     */
    Network::Network(const vector<size_t> &layers, const math::Func &activationFunc, const Layer::weights_initializer &init) : 
        inputSize(layers[0]), errorFunc(shared_ptr<error_func>{new error_func{mse}}){
        size_t oldSize = layers[0]; 

        for(size_t i = 1; i < layers.size(); i++){
            size_t numOfNeurons = layers[i];
            this->layers.push_back(Layer(numOfNeurons, activationFunc, this->layers.size()));
            this->layers.back().setWeights(init, oldSize);
            oldSize = numOfNeurons;
        }
    }

    /**
     * @brief Creates a fully connected network with random weights in the range specified by the hyper
     *        parameters. Every layer has its own activation function. The input size is the size of
     *        the first layer, the output size is the one of the last layer.
     *        NB: the input layer has no activation function, so the activationFuncs vector must be shorter
     *            than the layers one by 1.
     *
     * @param layers - The vector with the size of each layer.
     * @param activationFuncs - The activation function for each layer except the first.
     */
    Network::Network(const vector<size_t> &layers, const vector<math::Func> &activationFuncs, 
        const Layer::weights_initializer &init) : errorFunc(shared_ptr<error_func>{new error_func{mse}}){
        if(layers.size() - 1 != activationFuncs.size())
            throw invalid_argument("The sizes of layers and activation function vectors do not match.");

        this->inputSize = layers[0]; // Store the input size due to correctness check.

        size_t oldSize = layers[0];

        for(size_t i = 1; i < layers.size(); i++){
            this->layers.push_back(Layer(layers[i], activationFuncs[i-1], this->layers.size()));
            this->layers.back().setWeights(init, oldSize);
            oldSize = layers[i];
        }
    }

    /**
     * @brief Creates a new network equal to an existent one.
     *
     * @param net - The network to copy.
     */
    Network::Network(const Network &net) : layers(net.layers), inputSize(net.inputSize), errorFunc(net.errorFunc){ }

    /**
     * @brief Copy assignment.
     * 
     */
    Network& Network::operator = (const Network &rhs){
        this->layers = rhs.layers;
        this->inputSize = rhs.inputSize;
        this->errorFunc = rhs.errorFunc;

        return *this;
    }

    /**
     * @brief Move assignment.
     * 
     */
    Network& Network::operator = (Network &&rhs){
        this->layers = move(rhs.layers);
        this->inputSize = rhs.inputSize;
        this->errorFunc = move(rhs.errorFunc);

        return *this;
    }

    /**
     * @brief Sets the weights for each layer of the network.
     *
     * @param weights - The weights to set for each layer.
     */
    void Network::setWeights(const vector<vector<double>> &weights){
        if(weights.size() != this->layers.size())
            throw invalid_argument("The size of the weights vector and the number of layers do not agree.");

        for(size_t i = 0; i < this->layers.size(); ++i)
            this->layers[i].setWeights(weights[i]);
    }

    /**
     * @brief Sets the weights for each layer of the network.
     * 
     * @param weights - The weights to set.
     */
    void Network::setWeights(const vector<weightsMatrix> &weights){
        if(weights.size() != this->layers.size())
            throw invalid_argument("The size of the weights vector and the number of layers do not agree.");

        for(size_t i = 0; i < this->layers.size(); ++i)
            this->layers[i].setWeights(weights[i]);
    }

    /**
     * @brief Sets the weights for each layer of the network using move semantic.
     * 
     * @param weights - The weights to set.
     */
    void Network::setWeights(std::vector<weightsMatrix> &&weights){
        if(weights.size() != this->layers.size())
            throw invalid_argument("The size of the weights vector and the number of layers do not agree.");

        for(size_t i = 0; i < this->layers.size(); ++i)
            this->layers[i].setWeights(move(weights[i]));
    }

    /**
     * @brief Sets a new error function to minimize. This error function accepts in input the target value
     *        and the current output and returns the derivative of the error function.
     * 
     * @param error - The error function to minimize.
     */
    void Network::setErrorFunction(const error_func &error){
        this->errorFunc = shared_ptr<error_func>(new error_func(error));
    }

    /**
     * @brief Returns a vector with the matrix of weights of each layer.
     * 
     * @return vector<weightsMatrix> - The vector with the weights matrix.
     */
    vector<weightsMatrix> Network::getWeights() const{
        vector<weightsMatrix> vec(this->layers.size());

        for(size_t i = 0; i < this->layers.size(); ++i)
            vec[i] = this->layers[i].getWeights();

        return vec;
    }

    /**
     * @brief Returns a vector with the size of each layer.
     * 
     * @return vector<size_t> - A vector with the size of each layer.
     */
    vector<size_t> Network::getlayersSizes() const{ 
        vector<size_t> sizes(this->layers.size() + 1);

        sizes[0] = this->inputSize;
        for(size_t i = 0; i < this->layers.size(); ++i)
            sizes[i + 1] = this->layers[i].getSize();

        return sizes;
    } 

    // COMPUTATION

    /** 
     * @brief Computes the result for the given inputs.
     *
     * @param inputs - The inputs of the network.
     * @return std::vector<double> - The outputs computed by the network.
     */
    vector<double> Network::compute(const vector<double> &inputs){
        if(inputs.size() != this->inputSize)
            throw invalid_argument("The inputs size does not match the expected one.");

        vector<double> output(inputs);

        for(size_t i = 0; i < this->layers.size(); ++i)
            output = this->layers[i].feed_forward(output);

        return output;
    }

    // TRAIN

    /**
     * @brief The train step for a single pattern.
     *
     * @param trainPattern - The pattern of the training set.
     * @param expectedResults - The expected result.
     * @param est - The Estimator for the training set.
     */
    void Network::trainStep(const vector<double> &trainPattern, const vector<double> &expectedResults, Estimator &est){
        if(trainPattern.size() != this->inputSize)
            throw invalid_argument("The train pattern size does not match the input one.");

        vector<vector<double>> outputs = {trainPattern};
        outputs.reserve(this->layers.size() + 1);

        // Feed forward.
        for(size_t i = 0; i < this->layers.size(); ++i)
            outputs.push_back(this->layers[i].feed_forward(outputs.back()));

        auto results = outputs.back();
        // Check if the expected results have the right size.
        if(results.size() != expectedResults.size())
            throw invalid_argument("The results size does not match the expected one.");

        est.update(results, expectedResults); // Update the estimator.
        
        // Compute output errors for back propagation.
        vector<double> errors = (*this->errorFunc)(expectedResults, results);

        // Compute the backward step.
        for(short i = this->layers.size() - 1; i >= 0; i--)
            errors = this->layers[i].back_propagation(outputs[i], errors);
    }

    /**
     * @brief Trains the network using the training set passed as input.
     *
     * @param trainingSet - The training set.
     * @param est - The Estimator of the training set.
     * @param hyperPar - The hyperparameters.
     */
    void Network::train(const dataSet &trainingSet, Estimator &est, const parameters &hyperPar){
        if(trainingSet.inputs.size() != trainingSet.results.size())
            throw invalid_argument("The size of training set patterns and of the expected results do not match.");

        parameters currPars = hyperPar;
        size_t epoch;
        vector<vector<double>> myTrainPatterns(trainingSet.inputs);

        for(epoch = 0; epoch < hyperPar.max_epoch && !est.stoppingCriteria(); ++epoch){
            est.init(epoch);
            currPars.update(currPars, epoch); // Update the hyper-parameter.

            for(size_t i = 0; i < myTrainPatterns.size() / hyperPar.mb; ++i){
                auto end = i == myTrainPatterns.size() / hyperPar.mb - 1 ?
                            myTrainPatterns.size() : ((i + 1) * hyperPar.mb);

                // Compute the back propagation step for a group of patterns.
                for(size_t j = i * hyperPar.mb; j < end; ++j)
                    this->trainStep(myTrainPatterns[j], trainingSet.results[j], est);

                // Update the weights.
                for(size_t j = 0; j < this->layers.size(); ++j)
                    this->layers[j].updateWeights(hyperPar);
            }

            est.plot();
        }

        est.terminate();
    }

    /**
     * @brief Trains the network using the training set passed as input.
     *
     * @param trainingSet - The training set.
     * @param testSet - The test set.
     * @param trainEst - The Estimator for the training set.
     * @param testEst - The Estimator for the test set.
     * @param hyperPar - The hyperparameters.
     */
    void Network::train(const dataSet &trainingSet, const dataSet &testSet, Estimator &trainEst, Estimator &testEst, 
                            const parameters &hyperPar){
        if(trainingSet.inputs.size() != trainingSet.results.size())
            throw invalid_argument("The size of training set patterns and of the expected results do not match.");

        parameters currPars = hyperPar;
        vector<vector<double>> trainPatt{trainingSet.inputs}, trainRes{trainingSet.results};
        size_t epoch, mb_size = currPars.mb <= trainPatt.size() ? currPars.mb : trainPatt.size();

        for(epoch = 0; epoch < currPars.max_epoch && !trainEst.stoppingCriteria(); ++epoch){
            trainEst.init(epoch); testEst.init(epoch);
            currPars.update(currPars, epoch); // Update the hyper-parameter.

            // Compute test errors and accuracy.
            for(size_t j = 0; j < testSet.inputs.size(); ++j){
                vector<double> res = this->compute(testSet.inputs[j]);
                testEst.update(res, testSet.results[j]);
            }

            // Check again the stopping criteria due to the fact that the test estimator could have change it.
            for(size_t i = 0; i < trainPatt.size() / mb_size; ++i){
                auto end = i == trainPatt.size() / mb_size - 1 ?
                            trainPatt.size() : ((i + 1) * mb_size);

                // Compute the back propagation step for a group of patterns.
                for(size_t j = i * mb_size; j < end; ++j)
                    this->trainStep(trainPatt[j], trainRes[j], trainEst);

                // Update the weights.
                for(size_t j = 0; j < this->layers.size(); ++j)
                    this->layers[j].updateWeights(currPars);
            }

            trainEst.plot(); testEst.plot();
        }

        trainEst.terminate(); testEst.terminate();
    }
}