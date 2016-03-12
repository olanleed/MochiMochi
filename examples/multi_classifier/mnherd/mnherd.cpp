#include "../../../src/classifier/multi/mnherd.hpp"
#include "../../../src/utility/load_svmlight_file.hpp"
#include <boost/program_options.hpp>
#include <iostream>

int main(const int ac, const char* const * const av) {
  using namespace boost::program_options;

  options_description description("options");
  description.add_options()
    ("help", "")
    ("dim", value<int>()->default_value(0), "データの次元数")
    ("class", value<std::size_t>()->default_value(0), "クラス数")
    ("train", value<std::string>()->default_value(""), "学習データのファイルパス")
    ("test", value<std::string>()->default_value(""), "評価データのファイルパス")
    ("c", value<double>()->default_value(0.5), "ハイパパラメータ(C)")
    ("diagonal", value<int>()->default_value(0), "Diagonal Covariance, 0:Full 1:Exact 2:Project 3:Drop");

  variables_map vm;
  store(parse_command_line(ac, av, description), vm);
  notify(vm);

  if(vm.count("help")) { std::cout << description << std::endl; }

  const auto dim = vm["dim"].as<int>();
  const auto n_class = vm["class"].as<std::size_t>();
  const auto train_path = vm["train"].as<std::string>();
  const auto test_path = vm["test"].as<std::string>();
  const auto c = vm["c"].as<double>();
  const auto diagonal = vm["diagonal"].as<int>();

  std::string line;
  std::ifstream train_data(train_path);

  MNHERD mnherd(dim, n_class, c, diagonal);
  std::cout << "training..." << std::endl;
  while(std::getline(train_data, line)) {
    auto data = utility::read_ones(line, dim);
    mnherd.update(data.second, data.first);
  }

  int collect = 0;
  int all = 0;
  std::ifstream test_data(test_path);
  std::cout << "predicting..." << std::endl;
  while(std::getline(test_data, line)) {
    auto data = utility::read_ones(line, dim);
    auto pred = mnherd.predict(data.second);
    if(pred == data.first) {
      ++collect;
    }
    ++all;
  }

  std::cout << "Accuracy = " << (100.0 * collect / all) << "% (" << collect << "/" << all << ")" << std::endl;

  return 0;
}
