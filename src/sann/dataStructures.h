/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  DATA STRUCUTRES                                    *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/
#ifndef S_DATASTRUCTURES_S
#define S_DATASTRUCTURES_S

#include <string>
#include <vector>
#include <functional>
#include <iostream>

namespace sann{

// DATA STRUCTURE
typedef struct ds{
    std::vector<std::string> names; 
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> results;

    struct ds operator+(const struct ds x) const{
        struct ds newDs{names, inputs, results};
        newDs.names.insert(newDs.names.end(), x.names.begin(), x.names.end());
        newDs.inputs.insert(newDs.inputs.end(), x.inputs.begin(), x.inputs.end());
        newDs.results.insert(newDs.results.end(), x.results.begin(), x.results.end());
        return newDs;
    }

    /**
     * @brief Creates a new dataset from a subset of a given dataset. The dataset on which this method is call
     *        will be modified erasing the new sub-dataset.
     * 
     * @param start - The start index.
     * @param end - The end index. If it is > of the dataset size, the latter will be considered instead.
     * @return sann::dataSet - The new sub-dataset.
     */
    struct ds extractData(const size_t start, const size_t end){
        auto starti = this->inputs.begin() + start, startr = this->results.begin() + start,
            endi = this->inputs.begin() + end, endr = this->results.begin() + end;
        auto startn =  this->names.begin() + start, endn =  this->names.begin() + end;
            
        // Check if the size is right.
        if(end >= this->inputs.size()){
            endi = this->inputs.end(); endr = this->results.end(); endn = this->names.end();
        }
        
        // Build and erase the new dataset.
        std::vector<std::vector<double>> newInputs{starti, endi}, newResults{startr, endr};
        std::vector<std::string> newNames{startn, endn};
        this->inputs.erase(starti, endi); this->results.erase(startr, endr); this->names.erase(startn, endn);

        return {newNames, newInputs, newResults};
    }
} dataSet;


typedef struct p{
    std::size_t max_epoch, mb;
    float eta;
    float mi;
    float lambda;
    std::function<void(struct p &par, const size_t epoch)> update;
} parameters;

typedef std::vector<std::vector<double>> weightsMatrix;

/****************************************FUNCTIONS****************************************/


// Overload for << stream operator on weightsmatrix.
// std::ostream& operator<<(std::ostream& os, const weightsMatrix& weights)  
// {  
//     for(size_t i = 0; i < weights.size(); ++i){
//         for(size_t j = 0; j < weights[i].size(); ++j)
//             os << weights[i][j] << ' ';  
//         os << '\n';
//     }

//     return os;  
// }  

}  

#endif