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
#include "functions/enumerate.hpp"

class PA {
private :
  const std::size_t kDim;
  const double kC;
  const int kSelect;

private :
  Eigen::VectorXd _weight;

public :
  // int select : switching the PA algorithm
  // 0 : PA
  // 1 : PA-1
  // 2 : PA-2
  PA(const std::size_t dim, const double C, const int select = 2)
    : kDim(dim),
      kC(C),
      kSelect(select),
      _weight(Eigen::VectorXd::Zero(dim)) {

    static_assert(std::numeric_limits<decltype(dim)>::max() > 0, "Dimension Error. (Dimension > 0)");
    static_assert(std::numeric_limits<decltype(C)>::max() > 0, "Hyper Parameter Error. (C > 0)");
    assert(dim > 0);
    assert(C > 0);
    assert(select >= 0 && select <= 2);
  }

  virtual ~PA() { }

private :

  double suffer_loss(const Eigen::VectorXd& x, const int y) const {
    return std::max(0.0, 1.0 - y * _weight.dot(x));
  }

  double compute_margin(const Eigen::VectorXd& x) const {
    return _weight.dot(x);
  }

  double pa(const double value, const double loss) const {
    return loss / std::pow(std::abs(value), 2);
  }

  double pa_1(const double value, const double loss) const {
    return std::min(kC, pa(value, loss));
  }

  double pa_2(const double value, const double loss) const {
    return loss / (std::pow(std::abs(value), 2) + 1.0 / 2 * kC );
  }

  double compute_tau(const double value, const double loss) const {
    auto result = 0.0;
    switch(kSelect) {
    case 0 :
      result = pa(value, loss);
      break;
    case 1 :
      result = pa_1(value, loss);
      break;
    case 2 :
      result = pa_2(value, loss);
      break;
    default:
      std::runtime_error("Error in select number.");
    }
    return result;
  }

public :

  bool update(const Eigen::VectorXd& feature, const int label) {
    if (suffer_loss(feature, label) <= 0) { return false; }
    const auto loss = suffer_loss(feature, label);
    functions::enumerate(feature.data(), feature.data() + feature.size(), 0,
                         [&](const std::size_t index, const double value){
                           const auto tau = compute_tau(value, loss);
                           _weight[index] += tau * label * value;
                         });

    return true;
  }

  int predict(const Eigen::VectorXd& x) const {
    return compute_margin(x) > 0.0 ? 1 : -1;
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
