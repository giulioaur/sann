#ifndef BASE_ESTIMATOR
#define BASE_ESTIMATOR

#include <vector>
#include <string>
#include <iostream>
#include "../src/sann/constants.h"
#include "../src/sann/Estimator.hpp"
#include "../src/sann/Validator.hpp"
#include "../src/sann/math/Plotter.hpp"
#include "../src/sann/utility/FileManager.hpp"


#ifdef BASE_ESTIMATOR_USE_RELU
const double accuracyTreshold = 0.9;
#elif defined(BASE_ESTIMATOR_USE_LINEAR)
const double accuracyTreshold = 0;
#else
const double accuracyTreshold = 0;
#endif

// The error function
std::vector<double> errorFunction(const std::vector<double> &out, const std::vector<double> &expected);

// A set of estimators.

// A classic estimator with estimation of mse and accuracy.
class BaseEstimator : public sann::Estimator{ 
protected:
    double accuracy;
    double error = 1;
    std::size_t epoch, size;
    sann::math::Plotter plotter;
public:
    BaseEstimator(std::string plotFile) : plotter(plotFile){}

    void init(const std::size_t epoch){ this->accuracy = this->error = this->size = 0; this->epoch = epoch; }

    bool stoppingCriteria(){ return this->error == 0; }

    void update(const std::vector<double> &out, const std::vector<double> &expected){
        auto tmp = errorFunction(out, expected);  // Compute error and accuracy.
        this->error += tmp[0];                    // Update error.
        this->accuracy += tmp[1] == 0 ? 1 : 0;    // Update accuracy.
        ++size;                                   // Update number of samples.
    }

    void plot(){
        // Plot error and accuracy.
        this->error /= this->size; this->accuracy /= this->size;
        plotter.plotFunction({{(double)this->epoch}, {this->error}, {this->accuracy}});

        // Print the error every 100 epochs.
        if(this->epoch % 100 == 0)
            std::cout << std::to_string(this->epoch) << ": " << std::to_string(this->error) << std::endl;
    }

    void terminate(){ }

    double getAccuracy(){ return this->accuracy; }

    double getError(){ return this->error; }

    std::size_t getEpoch(){ return this->epoch; }
};

// VALIDATION PHASE.

// The training Estimator for model selection.
class BaseTrEstimator : public sann::Validator::TrValidEstimator{ 
private:
    std::size_t size = 0;
    std::string name;
public:
    BaseTrEstimator(const std::string &name) : TrValidEstimator(name), name(name){}

    std::unique_ptr<TrValidEstimator> clone(const std::string &filename) const {
        return std::unique_ptr<sann::Validator::TrValidEstimator>{new BaseTrEstimator{filename}};
    }

    std::string getName(){ return name; }

    void init(const std::size_t epoch){
        TrValidEstimator::init(epoch);
        this->size = 0;
    }

    void update(const std::vector<double> &out, const std::vector<double> &expected){
        auto tmp = errorFunction(out, expected);  // Compute error and accuracy.
        this->error += tmp[0];                    // Update error.
        this->accuracy += tmp[1] == 0 ? 1 : 0;    // Update accuracy.
        ++size;                                   // Update number of samples.
    }

    void finalize(){
        this->error /= this->size; this->accuracy /= this->size;
    }
};

// The validation set Estimator for the model selection.
class BaseVdEstimator : public sann::Validator::VdValidEstimator{ 
private:
    std::size_t size = 0;
    std::string filename;
    std::string result = "";
public:
    BaseVdEstimator(sann::Validator::TrValidEstimator &est) : VdValidEstimator(est){
        this->filename = dynamic_cast<BaseTrEstimator&>(est).getName();
        #ifdef BASE_ESTIMATOR_USE_RELU
        this->errorThreshold = 0.1;
        this->earlyThreshold = 300;
        #elif defined(BASE_ESTIMATOR_USE_LINEAR)
        this->errorThreshold = 2;
        this->earlyThreshold = 1000;
        #endif
    }

    std::unique_ptr<VdValidEstimator> clone(sann::Validator::TrValidEstimator &trEst) const {
        return std::unique_ptr<sann::Validator::VdValidEstimator>{new BaseVdEstimator{trEst}};
    }

    void init(const std::size_t epoch){
        VdValidEstimator::init(epoch);
        this->size = 0;
    }

    void update(const std::vector<double> &out, const std::vector<double> &expected){
        auto tmp = errorFunction(out, expected);  // Compute error and accuracy.
        this->error += tmp[0];                    // Update error.
        this->accuracy += tmp[1] == 0 ? 1 : 0;    // Update accuracy. 
        ++size;                                   // Update number of samples.
    }

    void finalize(){
        this->error /= this->size; this->accuracy /= this->size;
        this->result += std::to_string(this->getEpoch()) + ',' + std::to_string(error) + ',' + std::to_string(accuracy) + '\n';
    }

    void terminate(){
        sann::utility::FileManager::writeFile(sann::FILES_DIR + "validation/" + this->filename + ".vd.csv", this->result);
    }
};

std::vector<double> errorFunction(const std::vector<double> &out, const std::vector<double> &expected){
    std::vector<double> ret = {0, 0};

#ifdef BASE_ESTIMATOR_USE_MSE
    for(std::size_t i = 0; i < out.size(); ++i){
        double err = out[i] - expected[i];
        ret[0] += err * err;
        ret[1] += abs(err) < accuracyTreshold ? 0 : 1;
    }
#elif defined(BASE_ESTIMATOR_USE_MEE)
    for(std::size_t i = 0; i < out.size(); ++i){
        double err = out[i] - expected[i];
        ret[0] += err * err;
        ret[1] += abs(err) < accuracyTreshold ? 0 : 1;
    }
    ret[0] = sqrt(ret[0]);
#endif
    
    return ret;
}

#endif