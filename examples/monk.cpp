#define BASE_ESTIMATOR_USE_RELU     // The network has the relu as output layer.
#define BASE_ESTIMATOR_USE_MSE      // The network should be evaluated using mse.

#include <iostream>
#include <string>
#include "parse.hpp"
#include "../src/sann/math/Plotter.hpp"
#include "../src/sann/utility/FileManager.hpp"
#include "../src/sann/Regularizer.hpp"
#include "../src/sann/utility/Logger.hpp"
#include "../src/sann/utility/Stopwatch.hpp"
#include "../src/sann/constants.h"

using namespace std;
using namespace sann;
using namespace sann::math;
using namespace sann::utility;

void randomShuffle(dataSet &ds);

int main(int argc, char **argv){ 
    ///////////////////////////////////////MONK DATASET////////////////////////////////////////

    dataSet trainSet = FileManager::readDataSet(FILES_DIR + "dataSet/" + DATA_SET + ".train", 8, ' ', {0}, 7);
    dataSet testSet = FileManager::readDataSet(FILES_DIR + "dataSet/" + DATA_SET + ".test", 8, ' ', {0}, 7);
  
    dataSet trainKSet = Regularizer::getOneOfKDataSet(trainSet);
    dataSet testKSet = Regularizer::getOneOfKDataSet(testSet);
    
    // Create estimator
    BaseEstimator estTr{"trainErrors"}, estTe{"testErrors"};

    cout << "Start validation of " + DATA_SET + "..." << endl;
    cout << "Starting model selection" << endl;
    
    Stopwatch sw;
    
    ///////////////////////// Delete comment on this zone to try cross validation.////////////////////////////////
    // Clean tests folder.
    FileManager::cleanFolder(FILES_DIR + "validation");

    // Istantiate validator.
    Validator val = parse_validator(FILES_DIR + "config/" + DATA_SET + "_validation.json");

    randomShuffle(trainKSet);
    auto container = val.selectModelWithRisk(trainKSet, testKSet, estTr, estTe, 5);

    cout << "Risk: " << container.risk << endl;
    cout << "In training:" << estTr.getError() << " - " << estTr.getAccuracy() << endl;
    cout << "In test:" << estTe.getError() << " - " << estTe.getAccuracy() << endl;

    // Train the net with batch.
    // parameters hyperP;
    // Network net = parse_net(FILES_DIR + "config/config.json", hyperP);

    // net.train(trainKSet, testKSet, estTr, estTe, hyperP);

    cout << "Selection ended in " << sw.end() << endl;

    return 0;
}

void randomShuffle(dataSet &ds){
    size_t size = ds.inputs.size();
    unsigned int numOfShuffles = math::Randomizer::randomRange<unsigned int>(size * 2, size * 5);
    vector<size_t> swapVector = {};

    for(size_t i = 0; i < numOfShuffles; i++){
        int n1 = math::Randomizer::randomRange<int>(0, size),
            n2 = math::Randomizer::randomRange<int>(0, size);
        swap(ds.inputs[n1], ds.inputs[n2]);
        swap(ds.results[n1], ds.results[n2]);
    }
}