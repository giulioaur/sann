/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  ESTIMATOR CLASS HEADER                             *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/

#ifndef S_ESTIMATOR_S
#define S_ESTIMATOR_S

// System libraries include.
#include <vector>

// My includes.

namespace sann{

/// This class is the interface for the estimators used in the training phase. It needs to monitor the training 
/// phase wrt data.
/// This class has four important method:
/// - init: it is called at the beginning of each training iteration.
/// - stoppingCriteria: return true if the training has to be stop, false otherwise. NB: it is called before init.
/// - update: it is called at the end of training iteration of a single pattern.
/// - plot: it is called at the end of the epoch, after updating the weight.
/// - terminate: it is called once just before returning training method. 
class Estimator{
public:
    virtual void init(const std::size_t epoch) = 0;
    virtual bool stoppingCriteria() = 0;
    virtual void update(const std::vector<double> &out, const std::vector<double> &expected) = 0;
    virtual void plot() = 0;
    virtual void terminate() = 0;
};

}

#endif