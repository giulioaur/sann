/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  FILE MANAGER CLASS FILE                            *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/

#include "FileManager.hpp"

// Other system includes.
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace boost;

namespace sann {
namespace utility {

	/**
	 * @brief Creates a new folder.
	 * 
	 * @param folder - The name of the folder to create.
	 */
	void FileManager::createFolder(const std::string &folder){
		filesystem::path dir(folder);
		filesystem::create_directories(dir);
	}

	/**
	 * @brief Destroys a folder.
	 * 
	 * @param folder - The name of the folder to destroy.
	 */
	void FileManager::removeFolder(const std::string &folder){
		filesystem::path dir(folder);
		filesystem::remove_all(dir);
	}

	/**
	 * @brief Remove all contents of a folder but the folder itself.
	 * 
	 * @param folder - The folder to clean.
	 */
	void FileManager::cleanFolder(const string &folder){
		filesystem::path dir(folder);
		filesystem::remove_all(dir);
		filesystem::create_directory(dir);
	}
	
	/**
	 * @brief Returns the number of file in the folder.
	 * 
	 * @param folder - The name of the folder.
	 * @return size_t - The number of files inside it.
	 */
	size_t FileManager::getFilesNumber(const std::string &folder){
		filesystem::path dir(folder);
		return distance(filesystem::directory_iterator(dir), {});
	}

	/**
	 * @brief Returns the content of a file as a string.
	 * 
	 * @param fileName - The name of the file.
	 * @return string - Its content,
	 */
	string FileManager::flatTextFile(const std::string &fileName){
		ifstream file(fileName);
		if (file.good()) {
			stringstream buffer;

			buffer << file.rdbuf();
			return buffer.str();
		}
		throw std::ios_base::failure("File not found");
	}

	/**
	 * @brief Reads a dataset from a file. The datased must be encoded as csv file.
	 * 
	 * @param fileName - The name of the csv file.
	 * @param cols - The number of columns.
	 * @param separator - The column separator.
	 * @param resultCol - The index of the column in which is stored the result.
	 * @param nameCol - The index of the column in which is stored the name of the inputs.
	 * @return sann::dataSet - The dataset parsed.
	 */
	sann::dataSet FileManager::readDataSet(const string &fileName, const size_t cols, const char separator, 
			const vector<short> resultCols, const short nameCol){
		ifstream file(fileName); 

		if (file.good()) {
			sann::dataSet dataSet;
			short id = 0;
			string line;
				
			// Read line by line.
			while(getline(file, line)){
				vector<double> in, out;

				if(!line.empty() && line[0] != '#'){ 	// Avoid comment.
					std::stringstream ss(line);
					size_t i = 0;
					string col;

					// Give a name to the pattern if no one is provided.
					if(nameCol == -1)	dataSet.names.push_back("DataSet" + id++);

					// Read col by col.
					while(getline(ss, col, separator)){
						if(i >= cols)	
							throw invalid_argument("Parsing dataset: The number of colums exceeds the given one.");
						else{
							if(col.empty())					--i;
							else if(find(resultCols.begin(), resultCols.end(), i) != resultCols.end())		
															out.push_back(stod(col));
							else if((short)i == nameCol)	dataSet.names.push_back(col);
							else							in.push_back(stod(col));
						} 
						++i;
					}

					if(i < cols)
						throw invalid_argument("Parsing dataset: The number of colums is lower than the given one.");

					dataSet.inputs.push_back(in);
					dataSet.results.push_back(out);
				}
			}
					
			return dataSet;
		}

		throw std::ios_base::failure("File not found");
	}

	/**
	 * @brief Write a string on a file. If the file does not exist, it will be created.
	 * 
	 * @param fileName - The name of the file.
	 * @param content - The string to write.
	 * @param append - True if the content must be appended, false to clear the file before writing.
	 */
	void FileManager::writeFile(const std::string &fileName, const std::string &content, const bool append){
		auto flag = append ? fstream::app : fstream::trunc;
		ofstream file(fileName, fstream::out | flag);

		file << content;

		file.close();
	}
}
}
