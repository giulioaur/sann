#define BASE_ESTIMATOR_USE_LINEAR   // The network has the linear function in the output layer.
#define BASE_ESTIMATOR_USE_MEE      // The network should be evaluated using mse.

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

int main(int argc, char **argv){ 
    if(argc <= 1){
        cerr << "Insert the number of time the risk estimation has to be executed on different split." << endl;
        return -1;
    }

    size_t numberOfRiskEstimation = atoi(argv[1]);

    ///////////////////////////////////////CUP DATASET////////////////////////////////////////
    dataSet fullTrainSet = FileManager::readDataSet(FILES_DIR + "dataSet/ML-CUP17-TR.csv", 13, ',', {11, 12}, 0);
    dataSet fullTestSet = FileManager::readDataSet(FILES_DIR + "dataSet/ML-CUP17-TS.csv", 11, ',', {}, 0);

    int testLength = fullTrainSet.inputs.size() * 20 / 100;

    cout << "Start validation of " + DATA_SET + "..." << endl;
    cout << "Starting model selection" << endl;

    Stopwatch sw;
    
    // Clean tests folder.
    FileManager::cleanFolder(FILES_DIR + "validation");

    // Istantiate validator.
    Validator val = parse_validator(FILES_DIR + "config/" + DATA_SET + "_validation.json", false);

    for(size_t i = 0; i < numberOfRiskEstimation; ++i){
        dataSet trainSet{fullTrainSet};
        int pivot = Randomizer::randomRange<int>(0, trainSet.inputs.size() - testLength - 1);
        dataSet testSet = trainSet.extractData(pivot, pivot + testLength);
        int pivot2 = Randomizer::randomRange<int>(0, trainSet.inputs.size() - testLength - 1);
        dataSet validSet = trainSet.extractData(pivot2, pivot2 + testLength);

        // Create estimator
        BaseEstimator estTr{"trainErrors"}, estTe{"testErrors"};

        // auto container = val.selectModelWithRisk(trainSet, testSet, estTr, estTe, 5);
        auto container = val.selectModelWithRisk(trainSet, validSet, testSet, estTr, estTe);

        cout << "Risk: " << container.risk << endl;

        // Plot the point
        Plotter plt("points");
        for(size_t j = 0; j < testSet.inputs.size(); ++j){
            auto res = container.model.compute(testSet.inputs[j]);
            plt.plotFunction({{res[0]}, {res[1]}, {testSet.results[j][0]}, {testSet.results[j][1]}});
        }
    
        cout << "In training:" << estTr.getError() << " - " << estTr.getAccuracy() << endl;
        cout << "In test:" << estTe.getError() << " - " << estTe.getAccuracy() << endl;
    }


    // // Train the net with batch.
    // parameters hyperP;
    // Network net = parse_net(FILES_DIR + "config/config.json", hyperP);

    // net.train(trainSet, testSet, estTr, estTe, hyperP);

    cout << "Selection ended in " << sw.end() << endl;
    return 0;
}