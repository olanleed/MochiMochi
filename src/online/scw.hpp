#ifndef SRC_ONLINE_SCW_HPP_
#define SRC_ONLINE_SCW_HPP_

#include <Eigen/Core>
#include <boost/math/special_functions/erf.hpp>
#include <cmath>

class SCW {
private :
  const int DIM;
  const double C;
  const double PHI;

private :
  Eigen::MatrixXd _sigma;
  Eigen::VectorXd _mu;

private :
  inline double cdf(const double x) {
    return 0.5 * (1.0 + boost::math::erf(x / std::sqrt(2.0)));
  }

public :
  SCW(const int dim, const double c, const double eta)
    : DIM(dim), C(c), PHI(cdf(eta)) {
    static_assert(std::numeric_limits<decltype(dim)>::max() > 0, "Dimension Error. (Dimension > 0)");
    static_assert(std::numeric_limits<decltype(c)>::max() > 0, "Hyper Parameter Error. (c > 0)");
    static_assert(std::numeric_limits<decltype(eta)>::max() > 0, "Hyper Parameter Error. (η > 0)");

    _sigma = Eigen::MatrixXd::Identity(DIM, DIM);
    _mu = Eigen::VectorXd::Zero(DIM);
  }

  virtual ~SCW() { }

  double suffer_loss(const Eigen::VectorXd& x, const int label) {
    return std::max(0.0, PHI * std::sqrt(x.transpose() * _sigma * x) - label * _mu.dot(x));
  }

  double calculate_alpha(const double m, const double n, const double v, const double ganma) {
    const auto numerator = -(2.0 * m * n + PHI * PHI * m * v) + ganma;
    const auto denominator = 2.0 * (n * n + n * v * PHI * PHI);
    return std::max(0.0, numerator / denominator);
  }

  double calculate_beta(const double alpha, const double v) {
    const auto u = std::pow(-alpha * v * PHI + 4.0 * v , 2.0) / 4.0;
    return alpha * PHI / (std::sqrt(u) + v * alpha * PHI);
  }

  void fit(const Eigen::VectorXd& x, const int label) {
    const double v = x.transpose() * _sigma * x;
    const auto m = label * (_mu.dot(x));
    const auto n = v + 1.0 / 2.0 * C;
    const auto ganma = PHI * std::sqrt(PHI * PHI * m * m * v * v + 4.0 * n * v * (n + v * PHI * PHI));
    const auto alpha = calculate_alpha(m, n, v, ganma);
    const auto beta = calculate_beta(alpha, ganma);

    if (suffer_loss(x, label) > 0.0) {
      _mu.noalias() += alpha * label * _sigma * x;
      _sigma -= beta * _sigma * x * x.transpose() * _sigma;
    }
  }

  int predict(const Eigen::VectorXd& x) {
    return _mu.dot(x) < 0.0 ? -1 : 1;
  }

};

#endif
