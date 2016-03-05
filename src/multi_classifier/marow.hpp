#ifndef MOCHIMOCHI_MAROW_HPP_
#define MOCHIMOCHI_MAROW_HPP_

#include <algorithm>
#include <vector>
#include <boost/range/irange.hpp>
#include "../binary_classifier/arow.hpp"

class MAROW {
private:
  const std::size_t kClass;

private:
  std::vector<AROW> _arows;

public:
  MAROW(const std::size_t dim, const std::size_t n_class, const double r)
    : kClass(n_class) {
    static_assert(std::numeric_limits<decltype(n_class)>::max() >= 2, "Class range Error. (n_class >= 2)");

    for (const auto i : boost::irange<std::size_t>(0, kClass)) {
      _arows.push_back(AROW(dim, r));
    }
  }

  virtual ~MAROW() { }

public:
  void update(const Eigen::VectorXd& feature, const std::size_t label) {
    functions::enumerate(_arows.begin(), _arows.end(), 0,
                         [&](const std::size_t index, const AROW& arow) {
                           const auto t = (label - 1 == index) ? 1 : -1;
                           _arows[index].update(feature, t);
                         });
  }


  std::size_t predict(const Eigen::VectorXd& feature) const {
    const auto argmax = std::max_element(_arows.begin(), _arows.end(),
                                         [&](const AROW& p1, const AROW& p2) {
                                           return p1.dot(feature) < p2.dot(feature);
                                         });
    return std::distance(_arows.begin(), argmax) + 1;
  }

};

#endif //MOCHIMOCHI_MAROW_HPP_
