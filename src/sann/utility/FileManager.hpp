/*******************************************************
 *                                                     *
 *  sann: Neural Network library                       *
 *                                                     *
 *  FILE MANAGER CLASS HEADER                          *
 *                                                     *
 *  Giulio Auriemma                                    *
 *                                                     *
 *******************************************************/

#ifndef S_UTILITY_FILEMANAGER_S
#define S_UTILITY_FILEMANAGER_S

// System libraries includes.
#include <string>
#include <vector>
#include <boost/filesystem.hpp>

// My includes.
#include "../dataStructures.h"

namespace sann {
namespace utility {
	
/// This class wraps the interaction with the filesystem offering high level methods to interact with it.
/// It exploits the boost filesystem to manage the folder.
class FileManager {

public:
	// METHODS

	static void createFolder(const std::string &folder);
	static void removeFolder(const std::string &folder);
	static void cleanFolder(const std::string &folder);
	static std::size_t getFilesNumber(const std::string &folder);
	static std::string flatTextFile(const std::string &fileName);
	static sann::dataSet readDataSet(const std::string &fileName, const size_t cols, const char separator, 
		const std::vector<short> resultCols = {}, const short nameCol = -1);
	static void writeFile(const std::string &filename, const std::string &content, const bool append = false);
};
}
}

#endif
