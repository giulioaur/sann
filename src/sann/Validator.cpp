/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  VALIDATOR CLASS FILE                               *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/
#include "Validator.hpp"

//Other system includes.
#include <stdexcept>
#include <math.h>

// My include
#include "utility/Logger.hpp"
#include "utility/FileManager.hpp"

#ifdef S_DEBUG_MODE_S
#include <iostream>
#endif

using namespace std;
using namespace sann::math;

namespace sann{
    size_t validationNum = 0; // To avoid name clashes on file creation.

    /**
     * @brief Creates a new Validator.
     * 
     * @param loss - The loss function. It is used to check how much error a model has wrt validation set.
     * @param trainingEst - The training estimator. It is never used by itself, since every time an estimator
     *                      for the training set is needed, the clone method is called.
     * @param validationEst - The validation estimator. It is never used by itself, since every time an estimator
     *                        for the validation set is needed, the clone method is called.
     */
    Validator::Validator(const loss_func &loss, const shared_ptr<TrValidEstimator> trainingEst, 
        const shared_ptr<VdValidEstimator> validationEst) : loss(loss), trainingEst(trainingEst), 
        validationEst(validationEst) {}

    /**
     * @brief Returns a new validator with the same attribute of the old one.
     * 
     * @param val - The validator to copy.
     */
    Validator::Validator(const Validator &val) : loss(val.loss), epochs({val.epochs}), taus({val.taus}),
        alphas({val.alphas}), lambdas({val.lambdas}), etas({val.etas}), nets({val.nets}),
        trainingEst(val.trainingEst), validationEst(val.validationEst){}

    /**
     * @brief Computes the expected risk approximating it to the empirical risk.
     * 
     * @param net - The network on which compute the risk.
     * @param vs - The validation set.
     * @return double - The empirical risk.
     */
    double Validator::expectedRisk(sann::Network &net, const sann::dataSet &vs) const{
        double risk = 0;

        for(size_t i = 0; i < vs.inputs.size(); i++){ 
            vector<double> res = net.compute(vs.inputs[i]);
            risk += this->loss(res, vs.results[i]);
        }

        return risk / vs.inputs.size();
    } 

    /**
     * @brief Adds a new set of value on which search for the best model.
     *        NB: When the etas parameters (for learning rate decay) are passed, the vector vals has to be
     *            composed by both eta_0 and eta_t alternated, i.e. in the even position there might be eta_0
     *            followed by its eta_t. This behaviour is due to prevent crazy and not useful etas coupling.
     * 
     * @param type - The type of parameter to add.
     * @param vals - The set of parameter's values.
     */
    void Validator::addModelSelectionParameters(const msParameter type, const vector<float> &vals){
        switch(type){
            case Validator::EPOCH:
                for(auto value : vals)  
                    this->epochs.push_back((size_t) value); 
                break;
            case Validator::TAU:
                this->taus.insert(this->taus.end(), vals.begin(), vals.end()); break;
            case Validator::ETAS:
                for(size_t i = 0; i < vals.size() - 1; i += 2)
                    this->etas.push_back({vals[i], vals[i+1]});
                break;
            case Validator::MOMENTUM:
                this->alphas.insert(this->alphas.end(), vals.begin(), vals.end()); break;
            case Validator::REGULARIZATION:
                this->lambdas.insert(this->lambdas.end(), vals.begin(), vals.end()); break; 
        }
    }

    /**
     * @brief Add nets structures for the testing. 
     * 
     * @param nets - The networks to add.
     */
    void Validator::addModelSelectionNetwork(const vector<Network> &nets){
        this->nets.insert(this->nets.end(), nets.begin(), nets.end());
    }

    /**
     * @brief Set a new weight initializer.
     * 
     * @param initializers - The new weight initializer. 
     */
    void Validator::addModelSelectionWeightInit(const vector<initializer> &initializers){
        this->initializers.insert(this->initializers.end(), initializers.begin(), initializers.end());
    }

    /**
     * @brief Set the number of time every net should be initialized again with new starting values. Each
     *        set of sv is computed using once of the functions setted with method addModelSelectionWeightInit().
     *        To be sure that all the functions are used at least once, set n > the number of functions added.
     *        The sv already setted in the net when passed to model selector will be used once.  
     * 
     * @param n - The number of initialization for each net.
     */
    void Validator::setRandomInit(const size_t n){
        this->initNum = n;
    }

    /**
     * @brief Create the name of the file in which store validation result. If the folder containing the file
     *        does not exist, create it.
     * 
     * @param net - The neural network.
     * @param hyperP - The hyperparameters.
     * @param etaDecay - The vector composed by tau, eta0 and etat.
     * @return string - The name of the file.
     */
    string Validator::getValidatorName(sann::Network &net, const sann::parameters &hyperP, const vector<double> &etaDecay) const{
        vector<size_t> sizes = net.getlayersSizes();
        auto no_trail = [](const double x){
            string str = to_string(x);
            str.erase ( str.find_last_not_of('0') + 1, string::npos );
            if(str.back() == '.')   str.erase( str.size() - 1, string::npos);
            return str;
        };
        string str = "";
        
        for(size_t i = 0; i < sizes.size(); ++i)    
            str += i == sizes.size() - 1 ? to_string(sizes[i]) : to_string(sizes[i]) + "," ;

        str += "- t:" + no_trail(etaDecay[0]) + ", e:" + no_trail(etaDecay[1]) + "|" + no_trail(etaDecay[2]) 
            + ", m:" + no_trail(hyperP.mi) + ", l:" + no_trail(hyperP.lambda);

        #pragma omp critical(nameGiver)
        {
            utility::FileManager::createFolder(FILES_DIR + "validation/" + str);
            str += "/" + to_string(validationNum++);
        }
        
        return str; 
    }

    /**
     * @brief Return the best hyperparameters found by searching on all the possible combination. The returned model
     *        has the maximum number of epochs set to the mean of the various training stopping epochs. This way it
     *        can prevent going in overfitting exploiting the early stop of various training instance.
     * 
     * @param net - The network on.
     * @param tr - The training set.
     * @param vs - The validation set.
     * @return Validator::model - The best hyperparameters found.
     */
    Validator::pars_container Validator::gridSearch(Network &net, const dataSet &tr, const dataSet &vs) const{
        pars_container bestModel;
        unsigned long currEpochs = 0; // It is needed to do the mean between epochs.
        
        #pragma omp parallel for collapse(4)
        for(size_t ip = 0; ip < this->epochs.size(); ++ip){
        for(size_t it = 0; it < this->taus.size(); ++it){
        for(size_t ie = 0; ie < this->etas.size(); ++ie){
        for(size_t ia = 0; ia < this->alphas.size(); ++ia){
        for(size_t il = 0; il < this->lambdas.size(); ++il){
            // Create the net and train it using chosen hyperparameters.
            float tau = this->taus[it], eta0 = this->etas[ie][0], etat = this->etas[ie][1];
            parameters hyperP = {this->epochs[ip], tr.inputs.size(), 0, this->alphas[ia], this->lambdas[il], 
                [tau, eta0, etat](parameters &pars, const size_t epoch){
                    float alfa = min((double)epoch / tau, 1.);
                    pars.eta = (1. - alfa) * eta0 + alfa * etat;
            }};
            Network searchNet{net}; 
            auto trEst = this->trainingEst->clone(this->getValidatorName(net, hyperP, {tau, eta0, etat}));
            auto vdEst = this->validationEst->clone(*trEst);

            searchNet.train(tr, vs, *trEst, *vdEst, hyperP);

            // Compute the risk and check if it is the lower one.
            pars_container currModel{hyperP, this->expectedRisk(searchNet, vs), trEst->getAccuracy(), trEst->getError(),
                                    tau, eta0, etat};
            utility::Logger::writeLog(to_string(currModel.valError) + " | " + to_string(tau) + " | " +
                to_string(eta0) + " | " + to_string(etat) + " | " + to_string(this->alphas[ia]) + " | " +
                to_string(this->lambdas[il]), utility::Logger::type::NONE, false);

            #pragma omp critical(updateMin)
            {
            currEpochs += trEst->getEpoch() + 1;
            if(currModel < bestModel)   bestModel = currModel;  
            }
        }}}}}

        currEpochs /= this->epochs.size() * this->taus.size() * this->etas.size() * this->alphas.size() * this->lambdas.size();
        bestModel.pars.max_epoch = currEpochs;
        return bestModel;
    }

    /**
     * @brief Searches for the best model using grid search. The grid search is repeated over different 
     *        initialization of different networks. 
     * 
     * @param tr - The training set.
     * @param vs - The validation set.
     * @param net - The net on which hase been found the best model. It is returned through reference. 
     * @return Validator::completeModel
     */
    Validator::pars_container Validator::modelSearch(const dataSet &tr, const dataSet &vs, Network &net) const{
        if(this->nets.size() == 0 || this->epochs.size() == 0 || this->taus.size() == 0 || this->etas.size() == 0 ||
            this->alphas.size() == 0 || this->lambdas.size() == 0)
            throw range_error("Some parameter has not been setted.");

        pars_container bestModel; 
        
        // Do a grid search for each net.
        for(size_t i = 0; i < this->nets.size(); ++i){
            Network currNet{this->nets[i]}; 
            vector<size_t> layersSizes = currNet.getlayersSizes();

            for(size_t j = 0; j < this->initNum; ++j){
                vector<weightsMatrix> weights; // Needed to return the net with correct starting values.

                // Choose new starting values.
                if(this->initializers.size() > 0 && j > 0){
                    const size_t initToUse = j % this->initializers.size();
                    weights = this->initializers[initToUse](layersSizes);
                    currNet.setWeights(weights);
                }
                else
                    weights = currNet.getWeights();
                
                // Search the model.
                pars_container currModel = this->gridSearch(currNet, tr, vs);

                if(currModel < bestModel){
                    bestModel = currModel; 
                    net = currNet;
                    net.setWeights(move(weights));
                }
            }
        }

        return bestModel;
    }

    /**
     * @brief Searchs for the best model using k-fold cross validation.
     * 
     * @param trSet - The set to divide.
     * @param setsNum - The number of sets in which divide the training set.
     * @param net - The net on which hase been found the best model. It is returned through reference. 
     * @return Validator::pars_container - The best model found.
     */
    Validator::pars_container Validator::modelCrossSearch(const sann::dataSet &trSet, const size_t setsNum, Network &net) const{
        const size_t step = trSet.inputs.size() / setsNum;
        pars_container bestModel;

        // Divide the tr in k sets and for each one do model selection.
        // #pragma omp parallel for
        for(size_t i = 0; i < setsNum; i++){
            dataSet train{trSet}, validation = train.extractData(i * step, (i+1) * step);
            pars_container currModel = this->modelSearch(train, validation, net);
            
            #pragma omp critical(crossUpdateMin)
            {
            if(currModel < bestModel)   bestModel = currModel;
            }
        }

        return bestModel;
    }

    /**
     * @brief Selects the best model using the parameters setted for the model selection.
     * 
     * @param tr - The training set.
     * @param vs - The validation set.
     * @param est - The estimator to use in the final training.
     * @return sann::Network - The best trained net. 
     */ 
    Network Validator::selectModel(const dataSet &tr, const dataSet &vs, Estimator &est) const{
        Network net;
        Validator::pars_container model = this->modelSearch(tr, vs, net);

        utility::Logger::writeLog("Selected parameters: \n" + to_string(model.valError) + " | " + to_string(model.pars.eta)
                        + " | " + to_string(model.pars.mi) + " | " + to_string(model.pars.lambda), utility::Logger::type::NONE, false);

        net.train(tr + vs, est, model.pars);
        return net;
    }

    /**
     * @brief Returns the best model using cross validation. The model selection is executed using the
     *        parameters setted with method setModelSelectionParameters().
     * 
     * @param trainingSet - The set to split.
     * @param trainingSet - The estimator to use in the final training.
     * @param numOfSet - The number of set, i.e. the iterations of the algorithm. 
     * @return Network - The best trained net.
     */
    Network Validator::selectModelWithCross(const dataSet &trainingSet, sann::Estimator &est, const size_t numOfSet) const{
        Network net;
        pars_container bestModel;

        bestModel = this->modelCrossSearch(trainingSet, numOfSet, net);
        net.train(trainingSet, est, bestModel.pars);

        return net;
    }

    /**
     * @brief Select a model and then compute the estimation risk using a test set.
     * 
     * @param tr - The training set.
     * @param vd - The validation set.
     * @param ts - The test set.
     * @param trEst - The training estimator.
     * @param tsEst - The test estimator.
     * @return Validator::container - A struct with the weight and the risk. 
     */
    Validator::container Validator::selectModelWithRisk(const dataSet &tr, const dataSet &vd, const dataSet &ts, 
        Estimator &trainEst, Estimator &testEst) const{
        Network net;

        // Search for the best model and train the net on both training and validation sets.
        pars_container model = this->modelSearch(tr, vd, net);
        model.pars.mb = tr.inputs.size() + vd.inputs.size();
        net.train(tr + vd, ts, trainEst, testEst, model.pars);

        utility::Logger::writeLog("Selected parameters: \nerror: " + to_string(model.valError) + ", tau: " + 
            to_string(model.tau) + ", eta0: " + to_string(model.eta0) + ", etat: " + to_string(model.etat) +  
            ", mi" + to_string(model.pars.mi) + ", lambda" + to_string(model.pars.lambda), utility::Logger::type::NONE, false);


        return {net, this->expectedRisk(net, ts)}; 
    }

    /**
     * @brief Select a model with k-fold cross validation and then compute the estimation risk using a test set.
     * 
     * @param tr - The training set to split.
     * @param ts - The test set on which compute the risk.
     * @param trainEst - The estimator for training set to use during final training.
     * @param testEst - The estimator for test set to use during final training.
     * @param numOfSet - The number of set in which the training set has to be splitted during k-fold cross validation.
     * @return Validator::container Validator::selectModelWithRisk - A container with the best trained model and
     *                                                               its risk.
     */
    Validator::container Validator::selectModelWithRisk(const dataSet &tr, const dataSet &ts, Estimator &trainEst, 
        Estimator &testEst, const size_t numOfSet) const{
        Network net;

        // Search for the best model and train the net on the whole training set.
        pars_container model = this->modelCrossSearch(tr, numOfSet, net);
        model.pars.mb = tr.inputs.size();
        net.train(tr, ts, trainEst, testEst, model.pars);

        utility::Logger::writeLog("Selected parameters: \nerror: " + to_string(model.valError) + ", tau: " + 
            to_string(model.tau) + ", eta0: " + to_string(model.eta0) + ", etat: " + to_string(model.etat) +  
            ", mi: " + to_string(model.pars.mi) + ", lambda: " + to_string(model.pars.lambda), utility::Logger::type::NONE, false);

        return {net, this->expectedRisk(net, ts)}; 
    }
}