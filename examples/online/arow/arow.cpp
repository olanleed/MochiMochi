#include "../../../src/online/arow.hpp"
#include <boost/program_options.hpp>
#include <iostream>
#include <map>
#include <fstream>
#include <sstream>

std::pair<int, Eigen::VectorXd> parse(int size, const std::string& line) {
  Eigen::VectorXd data = Eigen::VectorXd::Zero(size);
  std::stringstream ss(line);
  std::string str;
  int label;

  ss >> label;
  while(ss >> str) {
    int n = 0;
    unsigned int i;
    for (i = 0; i < size; ++i) {
      if(str[i] == ':') {
        ++i;
        break;
      }
      n = 10 * n + (str.c_str()[i] - '0');
    }
    std::stringstream tt(str.substr(i));
    double value;
    tt >> value;
    data(n - 1) = value;
  }

  return std::make_pair(label, data);
}

int main(const int ac, const char* const * const av) {
  using namespace boost::program_options;

  options_description description("options");
  description.add_options()
    ("help", "")
    ("dim", value<int>()->default_value(0), "データの次元数")
    ("train", value<std::string>()->default_value(""), "学習データのファイルパス")
    ("test", value<std::string>()->default_value(""), "評価データのファイルパス")
    ("r", value<double>()->default_value(0.5), "ハイパパラメータ(r)");

  variables_map vm;
  store(parse_command_line(ac, av, description), vm);
  notify(vm);

  if(vm.count("help")) { std::cout << description << std::endl; }

  const auto dim = vm["dim"].as<int>();
  const auto train_path = vm["train"].as<std::string>();
  const auto test_path = vm["test"].as<std::string>();
  const auto r = vm["r"].as<double>();

  std::string line;
  std::ifstream train_data(train_path);

  AROW arow(dim, r);
  std::cout << "training..." << std::endl;
  while(std::getline(train_data, line)) {
    std::pair< int, Eigen::VectorXd > data = parse(dim, line);
    arow.fit(data.second, data.first);
  }

  int collect = 0;
  int all = 0;
  std::ifstream test_data(test_path);
  std::cout << "predicting..." << std::endl;
  while(std::getline(test_data, line)) {
    std::pair< int, Eigen::VectorXd > data = parse(dim, line);
    int pred = arow.predict(data.second);
    if(pred == data.first) {
      ++collect;
    }
    ++all;
  }

  std::cout << "Accuracy = " << (100.0 * collect / all) << "% (" << collect << "/" << all << ")" << std::endl;

  return 0;
}
