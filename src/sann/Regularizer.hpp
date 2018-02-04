/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  REGULARIZER CLASS HEADER                           *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/
#ifndef S_REGULARIZER_S
#define S_REGULARIZER_S

// My includes
#include "dataStructures.h"

namespace sann{

/// This is the class for the regularization of the data.
class Regularizer{
public:
    static dataSet getOneOfKDataSet(const dataSet &dataSet);
    static std::vector<double> getOneOfKVector(const std::vector<double> &vec, const std::vector<short> &min, const std::vector<short> &max);
};

}

#endif