#include "../libraries/json.hpp"
#include "BaseEstimators.hpp"
#include "../src/sann/Network.hpp" 
#include "../src/sann/Validator.hpp"
#include "../src/sann/math/Randomizer.hpp"

using json = nlohmann::json;
using namespace std;
using namespace sann;
using namespace sann::math;

Validator parse_validator(const string &file, const bool isClass = true);
Network parse_net(const string &fileName, parameters &hyperP);
vector<weightsMatrix> randomWeights(const vector<size_t> &sizes);
vector<weightsMatrix> randomGaussianWeights(const vector<size_t> &sizes);
vector<weightsMatrix> randomGaussianWeightsWithSqrt(const vector<size_t> &sizes);
vector<double> mee(const vector<double> &targets, const vector<double> &out);

Validator parse_validator(const string &file, const bool isClass){
    // Create estimator for the validation.
    shared_ptr<Validator::TrValidEstimator> vtEst{new BaseTrEstimator{""}};
    shared_ptr<Validator::VdValidEstimator> vdEst{new BaseVdEstimator{*vtEst}};

    // Istantiate validator.
    Validator val(isClass ? [](const vector<double> &res, const vector<double> ex) -> double{
        if(res.size() != ex.size())
            throw invalid_argument("The size of the current result does not match the size of the expected result.");

        bool equal = true;

        for(size_t i = 0; equal && i < res.size(); i++)
            equal &= (ex[i] - res[i]) < 0.9 ;

        return equal ? 0 : 1;
    }:
    [](const vector<double> &res, const vector<double> ex) -> double{
        double part = 0;

        for(size_t i = 0; i < res.size(); i++)
            part += (res[i] - ex[i]) * (res[i] - ex[i]);
        
        return sqrt(part);
    }, vtEst, vdEst);

    // Take validation configuration.
    std::ifstream validConf(file);
    json vConf;
    validConf >> vConf;

    // Add networks.
    auto nets = vConf["nets"];

    for(size_t i = 0; i < nets.size(); ++i){
        vector<Func> funcVec = {};
        auto functions = nets[i]["functions"];

        for(size_t i = 0; i < functions.size(); ++i){
            string tmp = functions[i];
            if(tmp == "linear")         funcVec.push_back(Func::linear);
            else if(tmp == "sigmoid")   funcVec.push_back(Func::sigmoid);
            else if(tmp == "tanh")      funcVec.push_back(Func::tanH);
            else if(tmp == "relu")      funcVec.push_back(Func::ReLU);
        }

        Network net{nets[i]["layers"], funcVec, [](const size_t m, const size_t n){
            weightsMatrix weights(m);

            for(size_t i = 0; i < m; ++i)
                weights[i] = math::Randomizer::randomGaussianVector<double>(0, 1./sqrt(n-1), n);

            return weights;
        }};
        
        if(nets[i]["error"] == "mee")
            net.setErrorFunction(mee);

        val.addModelSelectionNetwork({net});
    }

    // Add initializers.
    val.setRandomInit(vConf["initializersNum"]);

    auto inits = vConf["initializers"];
    vector<Validator::initializer> initializers(inits.size());

    for(size_t i = 0; i < inits.size(); ++i){
        string currInit = inits[i];

        if(currInit == "random")            initializers[i] = randomWeights;
        else if(currInit == "gaussian")     initializers[i] = randomGaussianWeights;
        else if(currInit == "gaussianSqrt") initializers[i] = randomGaussianWeightsWithSqrt;
    }

    val.addModelSelectionWeightInit(initializers);
    
    // Add hyperParameters.
    val.addModelSelectionParameters(Validator::EPOCH, vConf["max_epoch"]);
    val.addModelSelectionParameters(Validator::TAU, vConf["tau"]);
    val.addModelSelectionParameters(Validator::MOMENTUM, vConf["momentum"]);
    val.addModelSelectionParameters(Validator::REGULARIZATION, vConf["regularization"]);
    
    vector<vector<float>> etas = vConf["eta"];
    for(auto eta : etas)
        val.addModelSelectionParameters(Validator::ETAS, eta); 

    return val;
}

Network parse_net(const string &fileName, parameters &hyperP){
    std::ifstream validConf(fileName);
    json conf;
    validConf >> conf;

    vector<Func> funcVec = {};
    auto functions = conf["functions"];

    for(size_t i = 0; i < functions.size(); ++i){
        string tmp = functions[i];
        if(tmp == "linear")         funcVec.push_back(Func::linear);
        else if(tmp == "sigmoid")   funcVec.push_back(Func::sigmoid);
        else if(tmp == "tanh")      funcVec.push_back(Func::tanH);
        else if(tmp == "relu")      funcVec.push_back(Func::ReLU);
    }

    Network net{conf["layers"], funcVec, [](const size_t m, const size_t n){
        weightsMatrix weights(m);

        for(size_t i = 0; i < m; ++i)
            weights[i] = math::Randomizer::randomGaussianVector<double>(0, 1./sqrt(n-1), n);

        return weights;
    }};

    if(conf["error"] == "mee")
        net.setErrorFunction(mee);

    hyperP = {conf["epochs"], conf["mb_size"], conf["learning_rate"], conf["momentum"], conf["L2"], 
                [](parameters &par, const size_t epoch){}};
    return net;
}

/**********************************WEIGHTS INITS FUNCTION**********************************/

vector<weightsMatrix> randomWeights(const vector<size_t> &sizes){
    vector<weightsMatrix> weights(sizes.size() - 1);

    for(size_t i = 0; i < weights.size(); ++i)
        for(size_t j = 0; j < sizes[i+1]; ++j)
            weights[i].push_back(math::Randomizer::randomRangeVector<double>(-0.5, 0.5, sizes[i]));
            

    return weights;    
}

vector<weightsMatrix> randomGaussianWeights(const vector<size_t> &sizes){
    vector<weightsMatrix> weights(sizes.size() - 1);

    for(size_t i = 0; i < weights.size(); ++i)
        for(size_t j = 0; j < sizes[i+1]; ++j)
            weights[i].push_back(math::Randomizer::randomGaussianVector<double>(0, 1./sizes[i], sizes[i]));

    return weights;  
}


vector<weightsMatrix> randomGaussianWeightsWithSqrt(const vector<size_t> &sizes){
    vector<weightsMatrix> weights(sizes.size() - 1);

    for(size_t i = 0; i < weights.size(); ++i)
        for(size_t j = 0; j < sizes[i+1]; ++j)
            weights[i].push_back(math::Randomizer::randomGaussianVector<double>(0, 1./sqrt(sizes[i]), sizes[i]));

    return weights;      
}

/**********************************ERROR FUNCTION**********************************/

vector<double> mee(const vector<double> &targets, const vector<double> &out){
    vector<double> errors (targets.size());
    double mee = 0;

    for(size_t i = 0; i < targets.size(); ++i){
        double diff = out[i] - targets[i];
        mee += diff * diff;
    }
    
    mee = sqrt(mee);

    for(size_t i = 0; i < targets.size(); ++i){
        errors[i] = (targets[i] - out[i]) / mee;
    }

    return errors;
}