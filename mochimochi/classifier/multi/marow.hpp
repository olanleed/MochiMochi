#ifndef MOCHIMOCHI_MAROW_HPP_
#define MOCHIMOCHI_MAROW_HPP_

#include <algorithm>
#include <unordered_map>
#include <boost/range/irange.hpp>
#include "../binary/arow.hpp"

class MAROW {
private:
  const std::size_t kClass;

private:
  std::unordered_map<std::size_t, AROW> _arows;

public:
  MAROW(const std::size_t dim, const std::size_t n_class, const double r)
    : kClass(n_class) {
    static_assert(std::numeric_limits<decltype(n_class)>::max() > 2, "Class range Error. (n_class > 2)");

    for (const auto i : boost::irange<std::size_t>(1, kClass + 1)) {
      _arows.insert(std::pair<std::size_t, AROW>(i, AROW(dim, r)) );
    }
  }

  virtual ~MAROW() { }

public:
  void update(const Eigen::VectorXd& feature, const std::size_t label) {
    for(auto& arow : _arows) {
      const auto t = (arow.first == label) ? 1 : -1;
      arow.second.update(feature, t);
    }
  }

  std::size_t predict(const Eigen::VectorXd& feature) const {
    return std::max_element(_arows.begin(), _arows.end(),
                            [&](const auto& p1, const auto& p2) {
                              return p1.second.get_means().dot(feature) < p2.second.get_means().dot(feature);
                            })->first;
  }

};

#endif //MOCHIMOCHI_MAROW_HPP_
