#ifndef SRC_ONLINE_SCW_HPP_
#define SRC_ONLINE_SCW_HPP_

#include <Eigen/Core>
#include <boost/math/special_functions/erf.hpp>
#include <cmath>
#include <cstdbool>
#include "utility.hpp"

class SCW {
private :
  const int kDim;
  const double kC;
  const double kPhi;

private :
  Eigen::VectorXd _covariances;
  Eigen::VectorXd _means;

private :
  inline double cdf(const double x) {
    return 0.5 * (1.0 + boost::math::erf(x / std::sqrt(2.0)));
  }

public :
  SCW(const int dim, const double c, const double eta)
    : kDim(dim),
      kC(c),
      kPhi(cdf(eta)),
      _covariances(Eigen::VectorXd::Ones(kDim)),
      _means(Eigen::VectorXd::Zero(kDim)) {

    static_assert(std::numeric_limits<decltype(dim)>::max() > 0, "Dimension Error. (Dimension > 0)");
    static_assert(std::numeric_limits<decltype(c)>::max() > 0, "Hyper Parameter Error. (c > 0)");
    static_assert(std::numeric_limits<decltype(eta)>::max() > 0, "Hyper Parameter Error. (η > 0)");

  }

  virtual ~SCW() { }

  double suffer_loss(const Eigen::VectorXd& x, const int label) {
    const auto confidence = calculate_confidence(x);
    return std::max(0.0, kPhi * std::sqrt(confidence) - label * _means.dot(x));
  }

  double calculate_alpha(const double m, const double n, const double v, const double ganma) {
    const auto numerator = -(2.0 * m * n + kPhi * kPhi * m * v) + ganma;
    const auto denominator = 2.0 * (n * n + n * v * kPhi * kPhi);
    return std::max(0.0, numerator / denominator);
  }

  double calculate_beta(const double alpha, const double v) {
    const auto u = std::pow(-alpha * v * kPhi + 4.0 * v , 2.0) / 4.0;
    return alpha * kPhi / (std::sqrt(u) + v * alpha * kPhi);
  }

  double calculate_confidence(const Eigen::VectorXd& f) {
    auto confidence = 0.0;
    utility::enumerate(f.data(), f.data() + f.size(), 0,
                       [&](const int index, const double value) {
                         confidence += _covariances[index] * value * value;
                       });
    return confidence;
  }

  bool update(const Eigen::VectorXd& x, const int label) {
    const auto v = calculate_confidence(x);
    const auto m = label * _means.dot(x);
    const auto n = v + 1.0 / 2.0 * kC;
    const auto ganma = kPhi * std::sqrt(kPhi * kPhi * m * m * v * v + 4.0 * n * v * (n + v * kPhi * kPhi));
    const auto alpha = calculate_alpha(m, n, v, ganma);
    const auto beta = calculate_beta(alpha, ganma);

    if (suffer_loss(x, label) <= 0.0) { return false; }

    utility::enumerate(x.data(), x.data() + x.size(), 0,
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

};

#endif
