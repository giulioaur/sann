/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  STOPWATCH CLASS FILE	                           *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/
#include "Stopwatch.hpp"

using namespace std::chrono;

namespace sann {
namespace utility {
	/**
	 * @brief Creates a Stopwatch and a new checkpoint.
	 * 
	 */
	Stopwatch::Stopwatch(){
		timeStart = high_resolution_clock::now();
	}
	
	/**
	 * @brief Creates a new checkpoint.
	 * 
	 */
	void Stopwatch::start() {
		this->timeStart = high_resolution_clock::now();
	}

	/**
	 * @brief Returns the passed time from the last checkpoint.
	 * 
	 * @return double Stopwatch::end - The passed time from the last checkpoint.
	 */
	double Stopwatch::end() const{
		duration<double> time = high_resolution_clock::now() - this->timeStart;
		return time.count();
	}
}
}

