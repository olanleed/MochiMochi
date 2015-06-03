#ifndef MOCHIMOCHI_ADAGRAD_RDA_HPP_
#define MOCHIMOCHI_ADAGRAD_RDA_HPP_

#include <Eigen/Dense>
#include "functions/enumerate.hpp"

class ADAGRAD_RDA {
private :
  const std::size_t kDim;
  const double kEta;
  const double kLambda;

private :
  std::size_t _timestep;
  Eigen::VectorXd _w;
  Eigen::VectorXd _h;
  Eigen::VectorXd _g;

public :
  ADAGRAD_RDA(const std::size_t dim, const double eta, const double lambda)
    : kDim(dim),
      kEta(eta),
      kLambda(lambda),
      _timestep(0),
      _w(Eigen::VectorXd::Zero(kDim)),
      _h(Eigen::VectorXd::Zero(kDim)),
      _g(Eigen::VectorXd::Zero(kDim)) {
    static_assert(std::numeric_limits<decltype(dim)>::max() > 0, "Dimension Error. (Dimension > 0)");
    static_assert(std::numeric_limits<decltype(eta)>::max() > 0, "Hyper Parameter Error. (eta > 0)");
    static_assert(std::numeric_limits<decltype(lambda)>::max() > 0, "Hyper Parameter Error. (lambda > 0)");
    assert(dim > 0);
    assert(eta > 0);
    assert(lambda > 0);
  }

  virtual ~ADAGRAD_RDA() { }

private :

  double calculate_margin(const Eigen::VectorXd& x) const {
    return _w.dot(x);
  }

  double suffer_loss(const Eigen::VectorXd& x, const int y) const {
    return std::max(0.0, 1.0 - y * _w.dot(x));
  }

public :

  bool update(const Eigen::VectorXd& feature, const int label) {
    if (suffer_loss(feature, label) <= 0.0) { return false; }

    _timestep++;
    functions::enumerate(feature.data(), feature.data() + feature.size(), 0,
                       [&](const int index, const double value) {
                         const auto gradiant = -label * value;
                         _g[index] += gradiant;
                         _h[index] += gradiant * gradiant;

                         const auto sign = _g[index] >= 0 ? 1 : -1;
                         const auto eta = kEta / std::sqrt(_h[index]);
                         const auto u = std::abs(_g[index]) / _timestep;

                         _w[index] = (u <= kLambda) ? 0.0 : -sign * eta * _timestep * (u - kLambda);
                       });
    return true;
  }

  int predict(const Eigen::VectorXd& x) const {
    return calculate_margin(x) > 0.0 ? 1 : -1;
  }

};

#endif //MOCHIMOCHI_ADAGRAD_RDA_HPP_
