#ifndef MOCHIMOCHI_AROW_HPP_
#define MOCHIMOCHI_AROW_HPP_

#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include "functions/enumerate.hpp"

class AROW {
private :
  const std::size_t kDim;
  const double kR;

private :
  Eigen::VectorXd _covariances;
  Eigen::VectorXd _means;

public :
  AROW(const std::size_t dim, const double r)
    : kDim(dim),
      kR(r),
      _covariances(Eigen::VectorXd::Ones(kDim)),
      _means(Eigen::VectorXd::Zero(kDim)) {

    static_assert(std::numeric_limits<decltype(dim)>::max() > 0, "Dimension Error. (Dimension > 0)");
    static_assert(std::numeric_limits<decltype(r)>::max() > 0, "Hyper Parameter Error. (r > 0)");
    assert(dim > 0);
    assert(r > 0);

  }

  virtual ~AROW() { }

private :

  double suffer_loss(const double margin, const int label) const {
    return margin * label;
  }

  double compute_margin(const Eigen::VectorXd& x) const {
    return _means.dot(x);
  }

  double compute_confidence(const Eigen::VectorXd& feature) const {
    auto confidence = 0.0;
    functions::enumerate(feature.data(), feature.data() + feature.size(), 0,
                       [&](const int index, const double value) {
                         confidence += _covariances[index] * value * value;
                       });
    return confidence;
  }

public :

  bool update(const Eigen::VectorXd& feature, const int label) {
    const auto margin = compute_margin(feature);

    if (suffer_loss(margin, label) >= 1.0) { return false; }

    const auto confidence = compute_confidence(feature);
    const auto beta = 1.0 / (confidence + kR);
    const auto alpha = std::max(0.0, 1.0 - label * margin) * beta;

    functions::enumerate(feature.data(), feature.data() + feature.size(), 0,
                       [&](const int index, const double value) {
                         const auto v = _covariances[index] * value;
                         _means[index] += alpha * label * v;
                         _covariances[index] -= beta * v * v;
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
    std::vector<double> covariances_vector(_covariances.data(), _covariances.data() + _covariances.size());
    std::vector<double> means_vector(_means.data(), _means.data() + _means.size());
    ar & boost::serialization::make_nvp("covariances", covariances_vector);
    ar & boost::serialization::make_nvp("means", means_vector);
    ar & boost::serialization::make_nvp("dimension", const_cast<std::size_t&>(kDim));
    ar & boost::serialization::make_nvp("r", const_cast<double&>(kR));
  }

  template <class Archive>
  void load(Archive& ar, const unsigned int version) {
    std::vector<double> covariances_vector;
    std::vector<double> means_vector;
    ar & boost::serialization::make_nvp("covariances", covariances_vector);
    ar & boost::serialization::make_nvp("means", means_vector);
    ar & boost::serialization::make_nvp("dimension", const_cast<std::size_t&>(kDim));
    ar & boost::serialization::make_nvp("r", const_cast<double&>(kR));
    _covariances = Eigen::Map<Eigen::VectorXd>(&covariances_vector[0], covariances_vector.size());
    _means = Eigen::Map<Eigen::VectorXd>(&means_vector[0], means_vector.size());
  }
};

#endif //MOCHIMOCHI_AROW_HPP_
