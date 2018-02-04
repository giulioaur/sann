/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  LAYER CLASS FILE                                   *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/

#include "Layer.hpp"

// Other system includes.
#include <stdexcept>

// Debug includes.
#ifdef S_DEBUG_MODE_S
#include <iostream>
#endif

using namespace std;

namespace sann{

    /**
     * @brief Creates an empty layer.
     * 
     */
    Layer::Layer() : level(0), neurons(0), lastNet({}), func(math::Func::sigmoid) { }

    /**
     * @brief Instantiate a layer with n neurons with the same activation function and no weights.
     * 
     * @param numOfNeurons - The number of neurons.
     * @param activationFunc - The activation function.
     * @param level - The level of the layer. -1 by dafault.
     */
    Layer::Layer(const size_t numOfNeurons, const math::Func &activationFunc, const short level) : 
        level(level), neurons(numOfNeurons), lastNet(vector<double>(numOfNeurons)), 
        func(activationFunc){
        this->weights.resize(numOfNeurons);
    }

    /**
     * @brief Create a new layer with the same number and type of perceptron of the existent one.
     * 
     * @param lay - The existent layer.
     */
    Layer::Layer(const Layer &lay) : level(lay.level), neurons(lay.neurons), weights(weightsMatrix{lay.weights}),
        currErrors(weightsMatrix{lay.currErrors}), prevErrors(weightsMatrix{lay.prevErrors}), 
        lastNet(vector<double>(neurons)), func(lay.func){}

    /**
     * @brief Sets the weight of the layer.
     * 
     * @param weights - The weights to which init the neurons. The size should be equal to 
     *                  #neuron_prev_layer * #neuron_curr_layer. 
     */
    void Layer::setWeights(const std::vector<double> &weights){
        if(weights.size() % neurons != 0)
            throw invalid_argument("The size of the weights vector does not match the expected one.");

        size_t step = weights.size() / neurons;
        for(size_t i = 0, cap = neurons; i < cap; i++)
            this->weights[i] = vector<double>(weights.begin() + step * i, 
                                                weights.begin() + step * (i+1));
        
        this->currErrors = weightsMatrix{neurons, vector<double>(step)};
        this->prevErrors = weightsMatrix{neurons, vector<double>(step)};
    }

    /**
     * @brief Sets the weights of the layer.
     * 
     * @param weights - The new weights.
     */
    void Layer::setWeights(const weightsMatrix &weights){
        if(weights.size() != this->neurons)
            throw invalid_argument("The size of the weights vector does not match the expected one.");

        this->weights = weights;
        this->currErrors = weightsMatrix{neurons, vector<double>(weights[0].size())};
        this->prevErrors = weightsMatrix{neurons, vector<double>(weights[0].size())};
    }

    /**
     * @brief Sets the new weights of the layer using move semantics.
     * 
     * @param weights - The new weights.
     */
    void Layer::setWeights(weightsMatrix &&weights){
        if(weights.size() != this->neurons)
            throw invalid_argument("The size of the weights vector does not match the expected one.");

        this->weights = move(weights);
        this->currErrors = weightsMatrix{neurons, vector<double>(this->weights[0].size())};
        this->prevErrors = weightsMatrix{neurons, vector<double>(this->weights[0].size())};
    }

    /**
     * @brief Sets the weights matrix using a user-defined function.
     * 
     * @param init - The function that generates a random weights matrix m x n.
     * @param n - The number of neurons of the previous layer.
     */
    void Layer::setWeights(const weights_initializer &init, const size_t n){
        auto newWeights = init(this->neurons, n + 1);

        if(newWeights.size() != this->neurons)
            throw invalid_argument("The size of the weights matrix does not match the expected one.");

        this->weights = move(newWeights);
        this->currErrors = weightsMatrix{neurons, vector<double>(weights[0].size())};
        this->prevErrors = weightsMatrix{neurons, vector<double>(weights[0].size())};
    }

    /**
     * @brief Returns the weights of the layer.
     * 
     * @return weightsMatrix - The weights.
     */
    const weightsMatrix& Layer::getWeights() const{
        return this->weights;
    }

    /**
     * @brief Returns the number of neurons.
     * 
     * @return size_t - The number of neurons
     */
    size_t Layer::getSize() const{
        return this->neurons;
    }

    /**
     * @brief Computes the net function for a given inputs value.
     * 
     * @param inputs - The inputs.
     * @return vector<double> - The vector of the result of net function for each neuron.
     */
    vector<double> Layer::computeNets(vector<double> inputs) const{
        vector<double> nets(neurons);

        for(size_t i = 0; i < neurons; ++i){
            for(size_t j = 0; j < inputs.size(); ++j)
                nets[i] += inputs[j] * this->weights[i][j];
            nets[i] += this->weights[i].back();
        }

        return nets;
    }

    /**
     * @brief Computes the vector of outputs of the current layer. When this method is call, the result of the
     *        net function is stored inside the object to do not recompute it later during back propagation. So
     *        be careful in introducing network parallelization (for small net is not even necessary since -O3
     *        optimization flag is sufficient).
     * 
     * @param inputs - The inputs from the previous layer.
     * @return std::vector<double> - The vector of outputs of the current layer.
     */
    vector<double> Layer::feed_forward(const vector<double> &inputs){
        vector<double> outputs; 
        outputs.reserve(neurons);
        this->lastNet = move(this->computeNets(inputs));

        // Compute the output for each neuron.
        for(size_t i = 0; i < neurons; ++i)
            outputs.push_back(this->func.call(this->lastNet[i]));

        return outputs;
    }

    /**
     * @brief Applies the algorithm of back propagation on the current layer. It's important noting that to avoid
     *        compute the error on this layer once this function would be called on previous layer, the error is
     *        computed in the current call and then passed as argument to the previous layer.
     * 
     * @param inputs - The inputs of previous layer.
     * @param errors - The errors of the next layer. On the output layer9 this is the vector of the output errors.
     * @return std::vector<double> - The vector of errors to propagate back to previous layer.
     */
    vector<double> Layer::back_propagation(const vector<double> &inputs, const vector<double> &errors){
        vector<double> layerErrors(inputs.size());

        for(size_t j = 0; j < neurons; ++j){
            double delta = this->func.derivative(this->lastNet[j]) * errors[j];

            // Compute the delta weights for each weight and for the bias.
            for(size_t i = 0; i < inputs.size(); ++i){
                this->currErrors[j][i] += delta * inputs[i];
                layerErrors[i] += delta * this->weights[j][i]; // Update the error of the current layer.
            }
            this->currErrors[j].back() += delta;
        }

        return layerErrors;
    }

    /**
     * @brief Updates the weigths of the current layer. Use the hyperparameters to compute how much the weight
     *          will change.
     * 
     * @param hyperP - The hyperparameters.
     */ 
    void Layer::updateWeights(const sann::parameters &hyperP){
        for(size_t i = 0; i < neurons; ++i){
            for(size_t j = 0; j < this->weights[i].size(); ++j){
                double regularization = j < this->weights[i].size() - 1 ? hyperP.lambda * this->weights[i][j] : 0;
                double dwi = hyperP.eta * (this->currErrors[i][j] / hyperP.mb) + 
                                hyperP.mi * this->prevErrors[i][j];
                
                this->weights[i][j] += dwi - regularization;
                this->prevErrors[i][j] = dwi;
                this->currErrors[i][j] = 0; // Reset the error.
            }
        }
    }
}