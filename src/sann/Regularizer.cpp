/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  REGULARIZER CLASS FILE                             *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/
#include "Regularizer.hpp"

// Other system libraries include.
#include <stdexcept>
#include <climits>
#include <numeric>
#include <cmath>

using namespace std;

namespace sann{
    
    vector<vector<double>> app(vector<vector<double>> res){
        vector<vector<double>> ret(res.size());
        for(size_t i = 0; i < res.size(); ++i)
            ret[i] = res[i][0] == 0 ? vector<double>{-0.9} : vector<double>{0.9};

        return ret;
    }


    /**
     * @brief Returns a data set with the 1-of-k representations of the data on the starting data set.
     * 
     * @param dataSet - The starting data set.
     * @return dataSet - The data set with 1-of-k representations.
     */
    dataSet Regularizer::getOneOfKDataSet(const dataSet &dataSet){
        auto newClassVector = [](const vector<vector<double>> &vectors) -> vector<vector<double>>{
            vector<vector<double>> classVector;

            // Find maxima and minima for inputs sets.
            vector<short> maxima(vectors[0].size(), SHRT_MIN), minima(vectors[0].size(), SHRT_MAX);
            for(auto vec : vectors){
                for(size_t i = 0; i < vec.size(); i++){
                    if(maxima[i] < vec[i])    maxima[i] = vec[i];
                    if(minima[i] > vec[i])    minima[i] = vec[i];
                }
            }

            // Assign to new data set the regularized input.
            for(auto vec : vectors)
                classVector.push_back(Regularizer::getOneOfKVector(vec, minima, maxima));

            return classVector;
        };

        sann::dataSet newDataSet;
        #pragma omp parallel sections
        {
        #pragma omp section
        newDataSet.inputs = newClassVector(dataSet.inputs);
        #pragma omp section
        newDataSet.results = app(dataSet.results); //newClassVector(dataSet.results);
        #pragma omp section
        newDataSet.names = vector<string>(dataSet.names);
        }

        return newDataSet;
    }

    /**
     * @brief Returns the vector with the 1-of-k representation of the starting vector.
     * 
     * @param vec - The starting vector.
     * @param max - The max value for every element of the vector.
     * @param min - The min value for every element of the vector.
     * @return vector<double> - The vector with the 1-of-k representation.
     */
    vector<double> Regularizer::getOneOfKVector(const vector<double> &vec, const vector<short> &min, const vector<short> &max){
        if(vec.size() != max.size() || vec.size() != min.size())
            throw invalid_argument("The sizes of the vector, maxes and mins do not match.");
        
        // Compute the 1-of-k vector.
        vector<double> ret;
        
        for(size_t i = 0; i < vec.size(); i++)
            for(short j = min[i]; j <= max[i]; j++)
                ret.push_back(j == vec[i] ? 1 : 0);

        return ret;
    }

}