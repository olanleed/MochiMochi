#ifndef SRC_ONLINE_ADAGRAD_RDA_HPP_
#define SRC_ONLINE_ADAGRAD_RDA_HPP_

#include <Eigen/Core>
#include <cmath>
#include <map>

class ADAGRAD_RDA {
private :
  const int DIM;
  const double LAMBDA;

private :
  const double ETA = 1.0;

private :
  Eigen::VectorXd _w;
  Eigen::VectorXd _g;
  Eigen::VectorXd _h;
  int _t = 0;

public :
  ADAGRAD_RDA(const int dim, const double lambda) : DIM(dim), LAMBDA(lambda) {
    _w = _g = _h = Eigen::VectorXd::Zero(DIM);
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
    _t++;
    for (std::size_t i = 0; i < DIM; i++) {
      const double g = -label * feature[i];

      _g[i] += g;
      _h[i] += g * g;

      const int sign = _g[i] >= 0 ? 1 : -1;
      const double eta = ETA / std::sqrt(_h[i]);
      const double u = std::abs(_g[i]) / _t;

      if (u <= LAMBDA) {
        _w[i] = 0.0;
      } else {
        _w[i] = -sign * eta * _t * (u - LAMBDA);
      }
    }

   }

  int predict(const Eigen::VectorXd& x) const {
    return calculate_margin(x) >= 0.0 ? 1 : -1;
  }

};

#endif //SRC_ONLINE_ADAGRAD_RDA_HPP_
