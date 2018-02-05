/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  VALIDATOR CLASS HEADER                             *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/
#ifndef S_VALIDATOR_S
#define S_VALIDATOR_S

// System include library.
#include <functional>
#include <vector>
#include <limits>
#include <cstddef>
#include <numeric>

// My includes.
#include "constants.h" 
#include "dataStructures.h"
#include "Network.hpp"
#include "utility/FileManager.hpp"

const double MAX_DOUBLE = std::numeric_limits<double>::max();

namespace sann{

/// This is the class for the validation phase. It is used to select a model or compute an estimation of the risk.
// It exploits grid search and cross validation for a complete validation phase.
class Validator{
public:
    //CLASSES
    
    /**
    * @brief Those classes are a base class for Estimators that exploit the early stop criteria. They are abstract
    * class, so they need to be extended by another class to be instantiated. The aim of this behaviour is to
    * allow the user to use different kind of error on different model selection. The error is the error computed
    * during the training, not the loss of the Validator, that is specified on creation.
    * They will be passed to the training method.
    * NB: The model selection is better done if the error plotted by those estimator and the one minimized by
    *     the networks is the same.

    * To extends this class the user have to implement three methods: update(), clone() and finalize(). 
    * The first one is described in the Estimator header file, the second one returns a copy of the Estimator
    * and is needed to hide the copy constructor, the last one is called right before starting the plot method
    * (that could not be overrided). 
    * The attribute errorTreshold could be changed to edit the threshold at which the training error has to 
    * stop.
    * Due to speed reason, the plot is done by writing on a file a string with all the csv data.
    */
    class TrValidEstimator : public Estimator{
    private:
        bool earlyStop = false;
        std::size_t epoch, oldEpoch;
        std::string filename, results = "", oldResults;
        double oldError, oldAccuracy;
    protected:
        double accuracy, error = 1, errorThreshold = 10e-7;
    public:
        TrValidEstimator(const std::string &filename) : filename(filename) {}
        virtual std::unique_ptr<TrValidEstimator> clone(const std::string &filename) const = 0;
        virtual void init(const std::size_t epoch){
            this->accuracy = this->error = 0; this->epoch = epoch;
            this->results += std::to_string(epoch) + ',';
        }
        bool stoppingCriteria(){ return this->error <= errorThreshold || earlyStop; }
        virtual void finalize() = 0;
        void plot(){
            this->finalize();
            this->results += std::to_string(error) + ',' + std::to_string(accuracy) + '\n';
        }
        void terminate(){
            if(this->earlyStop){
                utility::FileManager::writeFile(FILES_DIR + "validation/" + filename + ".csv", this->oldResults);
                this->epoch = this->oldEpoch;
            }
            else
                utility::FileManager::writeFile(FILES_DIR + "validation/" + filename + ".csv", this->results);
        }
        virtual double getAccuracy() const{ return this->earlyStop ? this->oldAccuracy : this->accuracy; }
        virtual double getError() const{ return this->earlyStop ? this->oldError : this->error; }
        virtual double getEpoch() const{ return this->earlyStop ? this->oldEpoch : this->epoch; }
        void setEarlyStop(){ this->earlyStop = true; }
        void saveResults(){ 
            this->oldResults = this->results; this->oldEpoch = this->epoch; 
            this->oldError = this->error; this->oldAccuracy = this->accuracy;    
        }
    };
    
    /** To implement the early stop criteria, another estimator is introduced for the validation set.
    * This estimator keep tracks of the error of the validation set, if this error does not decrease after a 
    * certain number of epochs, the estimator stop the training. To do that it sets the attribute earlyStop
    * of the training estimator to true.
    * The three methods update(), clone() and finalize() have to be implemented (read above).
    * Three attribute could be changed: earlyStep changes the number of step in which earlyStop criteria is 
    * checked, earlyThreshold changes the number of bad checking has to be done before set earlyStop to true,
    * errorThreshold is the difference between last saved error and the current one to be sure that we are 
    * going into overfitting.
    */
    class VdValidEstimator : public Estimator{
    private:
        TrValidEstimator &trEst;
        std::size_t currIteration = 0, epoch;
        double oldError = MAX_DOUBLE, oldAccuracy = MAX_DOUBLE;
    protected:
        std::size_t earlyStep = 2, earlyThreshold = 1000;
        double error, accuracy, errorThreshold = 0.2;
    public:
        VdValidEstimator(TrValidEstimator &trEst) : trEst(trEst){}
        virtual std::unique_ptr<VdValidEstimator> clone(TrValidEstimator &trEst) const = 0;
        virtual void init(const std::size_t epoch){
            this->accuracy = this->error = 0; this->epoch = epoch;
        }
        bool stoppingCriteria(){ return false; }
        virtual void finalize() = 0;
        void plot(){ 
            this->finalize();
            // Each n epoch check if the mse is decreasing. If so, init current iteration to 0.
            if(this->epoch % this->earlyStep == 0){
                if(this->error < this->oldError){
                    this->oldError = this->error; this->oldAccuracy = this->accuracy;
                    this->currIteration = 0;
                    this->trEst.saveResults();
                }
                else if(this->error - this->oldError > errorThreshold){
                    this->currIteration++;
                }
            }   

            // If the number of iteration since mse decresed is more than the threshold, set the
            // earlystop attribute of the training estimator to true.
            if(this->currIteration >= this->earlyThreshold) 
                this->trEst.setEarlyStop();
        }
        virtual void terminate(){ }
        size_t getEpoch(){ return this->epoch; }
    };

private:

    // ATTRIBUTES

    std::function<double(const std::vector<double>&, const std::vector<double>&)> loss;
    std::vector<std::function<std::vector<weightsMatrix> (std::vector<std::size_t>)>> initializers = {};
    std::vector<size_t> epochs = {};
    std::vector<float> taus = {}, alphas = {}, lambdas = {};
    std::vector<std::vector<float>> etas;
    std::vector<Network> nets = {};
    std::size_t initNum = 1;
    const std::shared_ptr<TrValidEstimator> trainingEst;
    const std::shared_ptr<VdValidEstimator> validationEst;

    // STRUCTS

    struct pars_container{
        parameters pars = {};
        double valError = MAX_DOUBLE, accuracy = 0, trainError = MAX_DOUBLE;
        float tau, eta0, etat;

        bool operator < (const pars_container& other){
            return valError < other.valError || (valError == other.valError && 
                    (accuracy > other.accuracy || (accuracy == other.accuracy && trainError < other.trainError)));
        }
    };

    // METHODS

    std::string getValidatorName(sann::Network &net, const sann::parameters &hyperP, const std::vector<double> &etaDecay) const;
    pars_container gridSearch(sann::Network &net, const sann::dataSet &tr, const sann::dataSet &vs) const;
    pars_container modelSearch(const sann::dataSet &tr, const sann::dataSet &vs, Network &net) const;
    pars_container modelCrossSearch(const sann::dataSet &trSet, const size_t sets, Network &net) const;

public:

    // TYPEDEF

    typedef std::function<double(const std::vector<double>&, const std::vector<double>&)> loss_func;
    typedef std::function<std::vector<weightsMatrix> (std::vector<std::size_t>)> initializer;

    // ENUMERATION

    enum msParameter{EPOCH, TAU, ETAS, MOMENTUM, REGULARIZATION};

    // STRUCT

    /**
     * @brief A struct returned by model assesment: it contains the model plus the risk.
     * 
     */
    struct container{
        Network model;
        double risk;
    };

    //CONSTRUCTORS

    // Default constructor
    Validator(const loss_func &loss, const std::shared_ptr<TrValidEstimator> trainingEst, 
                const std::shared_ptr<VdValidEstimator> validationEst);
    // Copy constructor.
    Validator(const Validator &val);

    // METHODS

    void addModelSelectionParameters(const Validator::msParameter type, const std::vector<float> &val);
    void addModelSelectionNetwork(const std::vector<Network> &nets);
    void addModelSelectionWeightInit(const std::vector<initializer> &initializers);
    void setRandomInit(const std::size_t n);
    double expectedRisk(sann::Network &net, const sann::dataSet &vs) const;
    Network selectModel(const sann::dataSet &tr, const sann::dataSet &vs, sann::Estimator &est) const;
    Network selectModelWithCross(const sann::dataSet &trainingSet, sann::Estimator &est, const std::size_t numOfSet = 4) const;
    // The difference between the two following methods is that the first one uses simple model selection, while
    // the second one uses the k-fold cross validation for the model selection.
    Validator::container selectModelWithRisk(const sann::dataSet &trainingSet, const sann::dataSet &validationSet, 
        const sann::dataSet &testSet, sann::Estimator &trainEst, sann::Estimator &testEst) const;
    Validator::container selectModelWithRisk(const sann::dataSet &trainingSet, const sann::dataSet &testSet, 
        sann::Estimator &trainEst, sann::Estimator &testEst, const size_t numOfSet) const;
};

}

#endif