#ifndef SRC_ONLINE_AROW_HPP_
#define SRC_ONLINE_AROW_HPP_

#include <Eigen/Dense>
#include <cstdbool>
#include "utility.hpp"

class AROW {
private :
  const int kDim;
  const double kR;

private :
  Eigen::VectorXd _covariances;
  Eigen::VectorXd _means;

public :
  AROW(const int dim, const double r)
    : kDim(dim),
      kR(r),
      _covariances(Eigen::VectorXd::Ones(kDim)),
      _means(Eigen::VectorXd::Zero(kDim)) {

    static_assert(std::numeric_limits<decltype(dim)>::max() > 0, "Dimension Error. (Dimension > 0)");
    static_assert(std::numeric_limits<decltype(r)>::max() > 0, "Hyper Parameter Error. (r > 0)");

  }

  virtual ~AROW() { }

  double suffer_loss(const double margin, const int label) const {
    return margin * label;
  }

  double calculate_margin(const Eigen::VectorXd& x) const {
    return _means.dot(x);
  }

  double calculate_confidence(const Eigen::VectorXd& f) const {
    auto confidence = 0.0;
    utility::enumerate(f.data(), f.data() + f.size(), 0,
                       [&](const int index, const double value) {
                         confidence += _covariances[index] * value * value;
                       });
    return confidence;
  }

  bool update(const Eigen::VectorXd& feature, const int label) {
    const auto margin = calculate_margin(feature);
    const auto confidence = calculate_confidence(feature);
    const auto beta = 1.0 / (confidence + kR);
    const auto alpha = std::max(0.0, 1.0 - label * margin) * beta;

    if (suffer_loss(margin, label) >= 1.0) { return false; }

    utility::enumerate(feature.data(), feature.data() + feature.size(), 0,
                       [&](const int index, const double value) {
                         const auto v = _covariances[index] * value;
                         _means[index] += alpha * label * v;
                         _covariances[index] -= beta * v * v;
                       });
    return true;
  }

  int predict(Eigen::VectorXd& x) const {
    return calculate_margin(x) > 0.0 ? 1 : -1;
  }

};

#endif //SRC_ONLINE_AROW_HPP_
