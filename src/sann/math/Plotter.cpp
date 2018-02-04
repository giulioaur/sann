/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  PLOTTER CLASS FILE                                 *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/
#include "Plotter.hpp"

// Other system includes.
#include <stdexcept>
#include <fstream>

// My includes
#include "../constants.h"

using namespace std;

namespace sann{     
namespace math {

    /**
     * @brief Builds a new Plotter. If a file with the same name as the one passed to the constructor exists,
     *        it will be cleaned if clean flag is up. 
     * 
     * @param plotName - The name of the file on which plot (.csv will be postponed).
     * @param clean - True if the file must be cleaned.
     */
    Plotter::Plotter(const string &plotName, const bool clean){
        this->plotName = plotName;
        
        if(plotName.size() > 0){
            auto opt = clean ? ios::out | ios::trunc : ios::out;
            
            // Clear file if they exist.
            if(fstream{FILES_DIR + this->plotName + ".csv"})
                ofstream file(FILES_DIR + this->plotName + ".csv", opt);
            
            if(fstream{FILES_DIR + this->plotName + ".points.csv"})
                ofstream file(FILES_DIR + this->plotName + ".points.csv", opt);
        }
    }

    // METHODS

    /**
     * @brief Plots a list of vector as columns of csv file.
     * 
     * @param list - The list of columns vector.
     */
    void Plotter::plotFunction(const std::initializer_list<vector<double>> &list) const{
        if(plotName.size() > 0){
            ofstream file(FILES_DIR + this->plotName + ".csv", ios::out | ios::app);

            for(auto vec = list.begin(); list.size() > 1 && vec < list.end() - 1; vec++)
                if(vec->size() != (vec + 1)->size())
                    throw invalid_argument("The sizes of vectors do not match.");
            
            for(size_t i = 0; i < list.begin()->size(); i++){
                for(auto vec = list.begin(); vec < list.end(); vec++){
                    file << (*vec)[i];
                    if(vec < list.end() - 1)    file << ",";
                }
                file << endl;
            }

            file.close();
        }        
    }

    /**
     * @brief Plots a bidimensional function as columns of csv file.
     * 
     * @param x - The x of the function.
     * @param y - The y of the function.
     */
    void Plotter::plotFunction(const std::vector<double> &x, const std::vector<double> &y) const{
        if(plotName.size() > 0){
            ofstream file(FILES_DIR + this->plotName + ".csv", ios::out | ios::app);

            if(x.size() != y.size())
                throw invalid_argument("The sizes of two sets do not match.");

            for(size_t i = 0; i < x.size(); i++){
                file << x[i] << "," << y[i] << endl;
            }

            file.close();
        }
    }

    /**
     * @brief Plots a bidimensional function as culumns of csv file.
     * 
     * @param x - The input of the function.
     * @param fnc - The function itself.
     */
    void Plotter::plotFunction(const std::vector<double> &x, std::function<double(double)> fnc) const{
        if(plotName.size() > 0){
            ofstream file(FILES_DIR + this->plotName + ".csv", ios::out | ios::app);

            for(size_t i = 0; i < x.size(); i++){
                file << x[i] << "," << fnc(x[i]) << endl;
            }

            file.close();
        }
    }

    /**
     * @brief Plots a set of bidimensional points and their class.
     * 
     * @param x - The x coordinate.
     * @param y - The y coordinate.
     * @param classes - The class.
     */
    void Plotter::plotPoints(const std::vector<double> &x, const std::vector<double> &y, const std::vector<short> &classes) const{
        if(plotName.size() > 0){
            ofstream file(FILES_DIR + this->plotName + ".points.csv", ios::out | ios::app);
            
            if(x.size() != y.size() || (classes.size() > 0 && classes.size() != x.size()))
                throw invalid_argument("The sizes of two sets do not match.");

            for(size_t i = 0; i < x.size(); i++){
                short currClass = classes.size() > 0 ? classes[i] : 0;
                file << x[i] << "," << y[i] << "," << currClass << endl;
            }

            file.close();
        }
    }
}
}