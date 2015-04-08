#ifndef SRC_ONLINE_ADAM_HPP_
#define SRC_ONLINE_ADAM_HPP_

#include <Eigen/Core>
#include <cassert>
#include <cinttypes>

class ADAM {
private :
  const int kDim;

private :
  const double kAlpha = 0.001;
  const double kBeta1 = 0.9;
  const double kBeta2 = 0.999;
  const double kEpsilon = 0.00000001;
  const double kLambda = 0.99999999;

private :
  Eigen::VectorXd _w;
  Eigen::VectorXd _m;
  Eigen::VectorXd _v;
  std::size_t _timestep;

public :
  ADAM(const int dim)
    : kDim(dim),
      _timestep(0) {

    assert(dim > 0);

    _w = _m = _v = Eigen::VectorXd::Zero(kDim);
  }

  double suffer_loss(const Eigen::VectorXd& x, const int y) const {
    return std::max(0.0, 1.0 - y * _w.dot(x));
  }

  double calculate_margin(const Eigen::VectorXd& x) const {
    return _w.dot(x);
  }

  void update(const Eigen::VectorXd& feature, const int label) {

    if (suffer_loss(feature, label) <= 0.0)
      return ;

    const Eigen::VectorXd gradiant = -label * feature;
    const double beta1_t = std::pow(kLambda, _timestep) * kBeta1;

    _timestep++;
    for (std::size_t i = 0; i < kDim; i++) {
      _m[i] = beta1_t * _m[i] + (1.0 - beta1_t) * gradiant[i];
      _v[i] = kBeta2 * _v[i] + (1.0 - kBeta2) * gradiant[i] * gradiant[i];
      const double m_t = _m[i] / (1.0 - std::pow(kBeta1, _timestep));
      const double v_t = _v[i] / (1.0 - std::pow(kBeta2, _timestep));
      _w[i] -= kAlpha * m_t / (std::sqrt(v_t) + kEpsilon);
    }
  }

  int predict(const Eigen::VectorXd& feature) const {
    return calculate_margin(feature) > 0.0 ? 1 : -1;
  }

};

#endif //SRC_ONLINE_ADAM_HPP_
