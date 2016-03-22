#ifndef MOCHIMOCHI_PA_HPP_
#define MOCHIMOCHI_PA_HPP_

#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include <functional>
#include "../../functions/enumerate.hpp"

class PA {
private :
  const std::size_t kDim;
  const double kC;
  const int kSelect;

private :
  Eigen::VectorXd _weight;
  std::function<double(double, double)> _compute_tau;

public :
  PA(const std::size_t dim, const double C, const int select = 2)
    : kDim(dim),
      kC(C),
      kSelect(select),
      _weight(Eigen::VectorXd::Zero(dim)) {

    static_assert(std::numeric_limits<decltype(dim)>::max() > 0, "Dimension Error. (Dimension > 0)");
    static_assert(std::numeric_limits<decltype(C)>::max() > 0, "Hyper Parameter Error. (C > 0)");

    // int select : switching the PA algorithm
    // 0 : PA
    // 1 : PA-1
    // 2 : PA-2
    switch(kSelect) {
    case 0 :
      _compute_tau = [](const auto value, const auto loss) {
        return loss / std::pow(std::abs(value), 2);
      };
      break;
    case 1 :
      _compute_tau = [=](const auto value, const auto loss) {
        return std::min(kC, pa(value, loss));
      };
      break;
    case 2 :
      _compute_tau = [=](const auto value, const auto loss) {
        return loss / (std::pow(std::abs(value), 2) + 1.0 / 2 * kC );
      };
      break;
    default:
      std::runtime_error("Error in the PA algorithm.");
    }

  }

  virtual ~PA() { }

private :

  double suffer_loss(const Eigen::VectorXd& x, const int y) const {
    return std::max(0.0, 1.0 - y * _weight.dot(x));
  }

  double compute_margin(const Eigen::VectorXd& x) const {
    return _weight.dot(x);
  }

public :

  bool update(const Eigen::VectorXd& feature, const int label) {
    const auto loss = suffer_loss(feature, label);
    functions::enumerate(feature.data(), feature.data() + feature.size(), 0,
                         [&](const std::size_t index, const double value){
                           const auto tau = _compute_tau(value, loss);
                           _weight[index] += tau * label * value;
                         });

    return true;
  }

  int predict(const Eigen::VectorXd& x) const {
    return compute_margin(x) > 0.0 ? 1 : -1;
  }

  Eigen::VectorXd get_weight(void) const {
    return _weight;
  }

  void save(const std::string& filename) {
    std::ofstream ofs(filename);
    assert(ofs);
    boost::archive::text_oarchive oa(ofs);
    oa << *this;
    ofs.close();
  }

  void load(const std::string& filename) {
    std::ifstream ifs(filename);
    assert(ifs);
    boost::archive::text_iarchive ia(ifs);
    ia >> *this;
    ifs.close();
  }

private :
  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();
  template <class Archive>
  void save(Archive& ar, const unsigned int version) const {
    std::vector<double> weight(_weight.data(), _weight.data() + _weight.size());
    ar & boost::serialization::make_nvp("weigth", weight);
    ar & boost::serialization::make_nvp("dimension", const_cast<std::size_t&>(kDim));
    ar & boost::serialization::make_nvp("C", const_cast<double&>(kC));
  }

  template <class Archive>
  void load(Archive& ar, const unsigned int version) {
    std::vector<double> weight;
    ar & boost::serialization::make_nvp("weight", weight);
    ar & boost::serialization::make_nvp("dimension", const_cast<std::size_t&>(kDim));
    ar & boost::serialization::make_nvp("C", const_cast<double&>(kC));
    _weight = Eigen::Map<Eigen::VectorXd>(&weight[0], weight.size());
  }
};

#endif //MOCHIMOCHI_PA_HPP_
