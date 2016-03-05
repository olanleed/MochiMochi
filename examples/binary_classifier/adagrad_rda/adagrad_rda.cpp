#include "../../../src/classifier/binary/adagrad_rda.hpp"
#include "../../../src/utility/load_svmlight_file.hpp"
#include <boost/program_options.hpp>
#include <iostream>

int main(const int ac, const char* const * const av) {
  using namespace boost::program_options;

  options_description description("options");
  description.add_options()
    ("help", "")
    ("dim", value<int>()->default_value(0), "データの次元数")
    ("train", value<std::string>()->default_value(""), "学習データのファイルパス")
    ("test", value<std::string>()->default_value(""), "評価データのファイルパス")
    ("eta", value<double>()->default_value(0.5), "ハイパパラメータ(eta)")
    ("lambda", value<double>()->default_value(0.000001), "ハイパパラメータ(λ)");

  variables_map vm;
  store(parse_command_line(ac, av, description), vm);
  notify(vm);

  if(vm.count("help")) { std::cout << description << std::endl; }

  const auto dim = vm["dim"].as<int>();
  const auto train_path = vm["train"].as<std::string>();
  const auto test_path = vm["test"].as<std::string>();
  const auto eta = vm["eta"].as<double>();
  const auto lambda = vm["lambda"].as<double>();

  std::string line;
  std::ifstream train_data(train_path);

  ADAGRAD_RDA rda(dim, eta, lambda);
  std::cout << "training..." << std::endl;
  while(std::getline(train_data, line)) {
    auto data = utility::read_ones(line, dim);
    rda.update(data.second, data.first);
  }

  int collect = 0;
  int all = 0;
  std::ifstream test_data(test_path);
  std::cout << "predicting..." << std::endl;
  while(std::getline(test_data, line)) {
    auto data = utility::read_ones(line, dim);
    int pred = rda.predict(data.second);
    if(pred == data.first) {
      ++collect;
    }
    ++all;
  }

  std::cout << "Accuracy = " << (100.0 * collect / all) << "% (" << collect << "/" << all << ")" << std::endl;

  return 0;
}
