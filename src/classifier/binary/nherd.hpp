#ifndef MOCHIMOCHI_NHERD_HPP_
#define MOCHIMOCHI_NHERD_HPP_

#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include "../../functions/enumerate.hpp"

class NHERD {
private :
  const std::size_t kDim;
  const double kC;
  const int kDiagonal;

private :
  Eigen::VectorXd _covariances;
  Eigen::VectorXd _means;

public :
  // int diagonal : switching the diagonal covariance
  // 0 : Full covariance
  // 1 : Exact covariance
  // 2 : Project covariance
  // 3 : Drop covariance
  NHERD(const std::size_t dim, const double C, const int diagonal = 0)
    : kDim(dim),
      kC(C),
      kDiagonal(diagonal),
      _covariances(Eigen::VectorXd::Ones(kDim)),
      _means(Eigen::VectorXd::Zero(kDim)) {

    static_assert(std::numeric_limits<decltype(dim)>::max() > 0, "Dimension Error. (Dimension > 0)");
    static_assert(std::numeric_limits<decltype(C)>::max() > 0, "Hyper Parameter Error. (C > 0)");
    assert(dim > 0);
    assert(C > 0);
    assert(diagonal >= 0 && diagonal <= 3);
  }

  virtual ~NHERD() { }

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

  double full_covariance(const double covariance, const double confidence, const double value) const {
    const auto v = covariance * value;
    return covariance - (v * v * (kC * kC * confidence + 2 * kC) / std::pow((1.0 + kC * confidence), 2));
  }

  double exact_covariance(const double covariance, const double confidence, const double value) const {
    return covariance / std::pow(1.0 + kC * value * value * covariance, 2);
  }

  double project_covariance(const double covariance, const double confidence, const double value) const {
    return 1.0 / ((1.0 / covariance) + (2 * kC + kC * kC * confidence) * value * value);
  }

  double drop_covariance(const double covariance, const double confidence, const double value) const {
    return covariance - (std::pow(covariance * value, 2) * (kC * kC * confidence + 2 * kC) / std::pow(1.0 + kC * confidence, 2));
  }

  double compute_covariance(const double covariance, const double confidence, const double value) const {
    auto result = 0.0;
    switch(kDiagonal) {
    case 0 :
      result = full_covariance(covariance, confidence, value);
      break;
    case 1 :
      result = exact_covariance(covariance, confidence, value);
      break;
    case 2 :
      result = project_covariance(covariance, confidence, value);
      break;
    case 3 :
      result = drop_covariance(covariance, confidence, value);
      break;
    default:
      std::abort();
    }
    return result;
  }

public :

  bool update(const Eigen::VectorXd& feature, const int label) {
    const auto margin = compute_margin(feature);

    if (suffer_loss(margin, label) >= 1.0) { return false; }

    const auto confidence = compute_confidence(feature);
    const auto alpha = std::max(0.0, 1.0 - label * margin) / (confidence + 1 / kC) ;

    functions::enumerate(feature.data(), feature.data() + feature.size(), 0,
                       [&](const std::size_t index, const double value) {
                         _means[index] += alpha * label * _covariances[index] * value;
                         _covariances[index] = compute_covariance(_covariances[index], confidence, value);
                       });
    return true;
  }

  int predict(const Eigen::VectorXd& x) const {
    return compute_margin(x) > 0.0 ? 1 : -1;
  }

  Eigen::VectorXd get_means(void) const {
    return _means;
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
    ar & boost::serialization::make_nvp("C", const_cast<double&>(kC));
  }

  template <class Archive>
  void load(Archive& ar, const unsigned int version) {
    std::vector<double> covariances_vector;
    std::vector<double> means_vector;
    ar & boost::serialization::make_nvp("covariances", covariances_vector);
    ar & boost::serialization::make_nvp("means", means_vector);
    ar & boost::serialization::make_nvp("dimension", const_cast<std::size_t&>(kDim));
    ar & boost::serialization::make_nvp("C", const_cast<double&>(kC));
    _covariances = Eigen::Map<Eigen::VectorXd>(&covariances_vector[0], covariances_vector.size());
    _means = Eigen::Map<Eigen::VectorXd>(&means_vector[0], means_vector.size());
  }
};

#endif //MOCHIMOCHI_NHERD_HPP_
