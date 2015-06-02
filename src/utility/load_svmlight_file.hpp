#ifndef MOCHIMOCHI_LOAD_SVMLIGHT_FILE_HPP_
#define MOCHIMOCHI_LOAD_SVMLIGHT_FILE_HPP_

#include <Eigen/Dense>
#include <fstream>
#include <sstream>

namespace utility {
  inline std::pair<int, Eigen::VectorXd> read_ones(std::string line, const std::size_t dim) {
    Eigen::VectorXd values = Eigen::VectorXd::Zero(dim);
    std::istringstream parsed_line(line);
    int label;
    parsed_line >> label;

    std::string token;
    while(parsed_line >> token) {
      std::string::size_type pos = token.find(":");
      token.replace(pos, 1, " ");
      std::istringstream iss(token);
      int number;
      double value;
      iss >> number >> value;
      values(number - 1) = value;
    }
    return std::make_pair(label, values);
  }
}

#endif //MOCHIMOCHI_LOAD_SVMLIGHT_FILE_HPP_
