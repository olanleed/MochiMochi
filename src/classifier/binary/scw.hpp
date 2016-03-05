#ifndef MOCHIMOCHI_SCW_HPP_
#define MOCHIMOCHI_SCW_HPP_

#include <Eigen/Dense>
#include <boost/math/special_functions/erf.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include "../../functions/enumerate.hpp"

class SCW {
private :
  const std::size_t kDim;
  const double kC;
  const double kPhi;

private :
  Eigen::VectorXd _covariances;
  Eigen::VectorXd _means;

private :
  inline double cdf(const double x) const {
    return 0.5 * (1.0 + boost::math::erf(x / std::sqrt(2.0)));
  }

public :
  SCW(const std::size_t dim, const double c, const double eta)
    : kDim(dim),
      kC(c),
      kPhi(cdf(eta)),
      _covariances(Eigen::VectorXd::Ones(kDim)),
      _means(Eigen::VectorXd::Zero(kDim)) {

    static_assert(std::numeric_limits<decltype(dim)>::max() > 0, "Dimension Error. (Dimension > 0)");
    static_assert(std::numeric_limits<decltype(c)>::max() > 0, "Hyper Parameter Error. (c > 0)");
    static_assert(std::numeric_limits<decltype(eta)>::max() > 0, "Hyper Parameter Error. (η > 0)");
    assert(dim > 0);
    assert(c > 0);
    assert(eta > 0);
  }

  virtual ~SCW() { }

private :

  double suffer_loss(const Eigen::VectorXd& f, const int label) const {
    const auto confidence = compute_confidence(f);
    return std::max(0.0, kPhi * std::sqrt(confidence) - label * _means.dot(f));
  }

  //Proposition 1
  double compute_alpha(const double m, const double n, const double v, const double ganma) const {
    const auto psi = 1.0 + kPhi * kPhi / 2.0;
    const auto zeta = 1.0 + kPhi * kPhi;
    const auto tmp1 = -m * psi + std::sqrt(m * m * std::pow(kPhi, 4.0) / 4.0 + v * kPhi * kPhi * zeta);
    const auto tmp2 = 1.0 / v * zeta * tmp1;
    return std::min(kC, std::max(0.0, tmp2));
  }

  double compute_beta(const double alpha, const double v) const {
    const auto u = std::pow(-alpha * v * kPhi + std::sqrt(alpha * alpha * v * v * kPhi * kPhi + 4.0 * v), 2.0) / 4.0;
    return alpha * kPhi / (std::sqrt(u) + v * alpha * kPhi);
  }

  double compute_confidence(const Eigen::VectorXd& f) const {
    auto confidence = 0.0;
    functions::enumerate(f.data(), f.data() + f.size(), 0,
                       [&](const int index, const double value) {
                         confidence += _covariances[index] * value * value;
                       });
    return confidence;
  }

public :

  bool update(const Eigen::VectorXd& feature, const int label) {
    const auto v = compute_confidence(feature);
    const auto m = label * _means.dot(feature);
    const auto n = v + 1.0 / 2.0 * kC;
    const auto ganma = kPhi * std::sqrt(kPhi * kPhi * m * m * v * v + 4.0 * n * v * (n + v * kPhi * kPhi));
    const auto alpha = compute_alpha(m, n, v, ganma);
    const auto beta = compute_beta(alpha, ganma);

    if (suffer_loss(feature, label) <= 0.0) { return false; }

    functions::enumerate(feature.data(), feature.data() + feature.size(), 0,
                       [&](const int index, const double value) {
                         const auto v = _covariances[index] * value;
                         _means[index] += alpha * label * v;
                         _covariances[index] -= beta * v * v;
                       });

    return true;
  }

  int predict(const Eigen::VectorXd& x) {
    return _means.dot(x) < 0.0 ? -1 : 1;
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
    ar & boost::serialization::make_nvp("phi", const_cast<double&>(kPhi));
    ar & boost::serialization::make_nvp("c", const_cast<double&>(kC));
  }

  template <class Archive>
  void load(Archive& ar, const unsigned int version) {
    std::vector<double> covariances_vector;
    std::vector<double> means_vector;
    ar & boost::serialization::make_nvp("covariances", covariances_vector);
    ar & boost::serialization::make_nvp("means", means_vector);
    ar & boost::serialization::make_nvp("dimension", const_cast<std::size_t&>(kDim));
    ar & boost::serialization::make_nvp("phi", const_cast<double&>(kPhi));
    ar & boost::serialization::make_nvp("c", const_cast<double&>(kC));
    _covariances = Eigen::Map<Eigen::VectorXd>(&covariances_vector[0], covariances_vector.size());
    _means = Eigen::Map<Eigen::VectorXd>(&means_vector[0], means_vector.size());
  }

};

#endif //MOCHIMOCHI_SCW_HPP_
