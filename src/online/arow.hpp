#ifndef SRC_ONLINE_AROW_HPP_
#define SRC_ONLINE_AROW_HPP_

#include <Eigen/Core>

class AROW {
private :
  const int DIM;
  const double R;

private :
  Eigen::MatrixXd _sigma;
  Eigen::VectorXd _mu;

public :
  AROW(const int dim, const double r) : DIM(dim), R(r) {
    static_assert(std::numeric_limits<decltype(dim)>::max() > 0, "Dimension Error. (Dimension > 0)");
    static_assert(std::numeric_limits<decltype(r)>::max() > 0, "Hyper Parameter Error. (r > 0)");

    _sigma = Eigen::MatrixXd::Identity(DIM, DIM);
    _mu = Eigen::VectorXd::Zero(DIM);
  }

  virtual ~AROW() { }

  double suffer_loss(const double margin, const int label) const {
    return margin * label;
  }

  double calculate_margin(const Eigen::VectorXd& x) const {
    return _mu.dot(x);
  }

  double calculate_confidence(const Eigen::VectorXd& x) const {
    return x.transpose() * _sigma * x;
  }

  void update(const Eigen::VectorXd& feature, const int label) {
    const auto margin = calculate_margin(feature);
    const auto confidence = calculate_confidence(feature);
    const auto beta = 1.0 / (confidence + R);
    const auto alpha = std::max(0.0, 1.0 - label * feature.transpose() * _mu) * beta;

    if (suffer_loss(margin, label) < 1.0) {
      _mu.noalias() += alpha * _sigma * label * feature;
      _sigma -= beta * _sigma * feature * feature.transpose() * _sigma;
    }
  }

  int predict(Eigen::VectorXd& x) const {
    return calculate_margin(x) > 0.0 ? 1 : -1;
  }

};

#endif //SRC_ONLINE_AROW_HPP_
