#ifndef MOCHIMOCHI_ADAM_HPP_
#define MOCHIMOCHI_ADAM_HPP_

#include <Eigen/Dense>
#include <cassert>
#include "functions/enumerate.hpp"

class ADAM {
private :
  const std::size_t kDim;

private :
  std::size_t _timestep;
  Eigen::VectorXd _w;
  Eigen::VectorXd _m;
  Eigen::VectorXd _v;

public :
  ADAM(const std::size_t dim)
    : kDim(dim),
      _timestep(0),
      _w(Eigen::VectorXd::Zero(kDim)),
      _m(Eigen::VectorXd::Zero(kDim)),
      _v(Eigen::VectorXd::Zero(kDim)) {

    assert(dim > 0);
  }

  virtual ~ADAM() { }

private :

  double suffer_loss(const Eigen::VectorXd& x, const int y) const {
    return std::max(0.0, 1.0 - y * _w.dot(x));
  }

  double calculate_margin(const Eigen::VectorXd& x) const {
    return _w.dot(x);
  }

public :

  bool update(const Eigen::VectorXd& feature, const int label) {
    constexpr auto kAlpha = 0.001;
    constexpr auto kBeta1 = 0.9;
    constexpr auto kBeta2 = 0.999;
    constexpr auto  kEpsilon = 0.00000001;
    constexpr auto kLambda = 0.99999999;

    if (suffer_loss(feature, label) <= 0.0) { return false; }

    const Eigen::VectorXd gradiant = -label * feature;
    const auto beta1_t = std::pow(kLambda, _timestep) * kBeta1;

    _timestep++;
    functions::enumerate(gradiant.data(), gradiant.data() + gradiant.size(), 0,
                       [&](const std::size_t index, const double value) {
                         _m[index] = beta1_t * _m[index] + (1.0 - beta1_t) * value;
                         _v[index] = kBeta2 * _v[index] + (1.0 - kBeta2) * value * value;
                         const auto m_t = _m[index] / (1.0 - std::pow(kBeta1, _timestep));
                         const auto v_t = _v[index] / (1.0 - std::pow(kBeta2, _timestep));
                         _w[index] -= kAlpha * m_t / (std::sqrt(v_t) + kEpsilon);
                       });

    return true;
  }

  int predict(const Eigen::VectorXd& feature) const {
    return calculate_margin(feature) > 0.0 ? 1 : -1;
  }

};

#endif //MOCHIMOCHI_ADAM_HPP_
