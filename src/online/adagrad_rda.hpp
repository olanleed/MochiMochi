#ifndef SRC_ONLINE_ADAGRAD_RDA_HPP_
#define SRC_ONLINE_ADAGRAD_RDA_HPP_

#include <Eigen/Core>
#include <cinttypes>
#include <cmath>

class ADAGRAD_RDA {
private :
  const int kDim;
  const double kEta;
  const double kLambda;

private :
  Eigen::VectorXd _w;
  Eigen::VectorXd _g;
  Eigen::VectorXd _h;
  std::size_t _timestep;

public :
  ADAGRAD_RDA(const int dim, const double eta, const double lambda)
    : kDim(dim),
      kEta(eta),
      kLambda(lambda),
      _timestep(0) {
    _w = _g = _h = Eigen::VectorXd::Zero(kDim);
  }

  virtual ~ADAGRAD_RDA() { }

  double calculate_margin(const Eigen::VectorXd& x) const {
    return _w.dot(x);
  }

  double suffer_loss(const Eigen::VectorXd& x, const int y) const {
    return std::max(0.0, 1.0 - y * _w.dot(x));
  }

  void update(const Eigen::VectorXd& feature, const int label) {
    if (suffer_loss(feature, label) <= 0.0) { return ; }

    _timestep++;
    for (std::size_t i = 0; i < kDim; i++) {
      const auto gradiant = -label * feature[i];
      _g[i] += gradiant;
      _h[i] += gradiant * gradiant;

      const int sign = _g[i] >= 0 ? 1 : -1;
      const double eta = kEta / std::sqrt(_h[i]);
      const double u = std::abs(_g[i]) / _timestep;

      if (u <= kLambda) {
        _w[i] = 0.0;
      } else {
        _w[i] = -sign * eta * _timestep * (u - kLambda);
      }
    }

   }

  int predict(const Eigen::VectorXd& x) const {
    return calculate_margin(x) > 0.0 ? 1 : -1;
  }

};

#endif //SRC_ONLINE_ADAGRAD_RDA_HPP_
