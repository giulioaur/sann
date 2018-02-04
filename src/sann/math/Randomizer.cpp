/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  RANDOMIZER CLASS HEADER                            *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/
#include "Randomizer.hpp"

//Other system libraries include.
#include <time.h>
#include <stdlib.h>

namespace sann{
namespace math{
    
    //Init a random seed.
    int Randomizer::initSeed = []() -> int { srand(time(NULL)); return 1; }();

}
}