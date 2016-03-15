#ifndef MOCHIMOCHI_MNHERD_HPP_
#define MOCHIMOCHI_MNHERD_HPP_

#include <algorithm>
#include <unordered_map>
#include <boost/range/irange.hpp>
#include "../binary/nherd.hpp"

class MNHERD {
private:
  const std::size_t kClass;

private:
  std::unordered_map<std::size_t, NHERD> _nherds;

public:
  MNHERD(const std::size_t dim, const std::size_t n_class, const double C, const int diagonal = 0)
    : kClass(n_class) {
    static_assert(std::numeric_limits<decltype(n_class)>::max() > 2, "Class range Error. (n_class > 2)");

    for (const auto i : boost::irange<std::size_t>(1, kClass + 1)) {
      _nherds.insert(std::pair<std::size_t, NHERD>(i, NHERD(dim, C, diagonal)) );
    }
  }

  virtual ~MNHERD() { }

public:
  void update(const Eigen::VectorXd& feature, const std::size_t label) {
    for(auto& nherd : _nherds) {
      const auto t = (nherd.first == label) ? 1 : -1;
      nherd.second.update(feature, t);
    }
  }

  std::size_t predict(const Eigen::VectorXd& feature) const {
    return std::max_element(_nherds.begin(), _nherds.end(),
                            [&](const auto& p1, const auto& p2) {
                              return p1.second.get_means().dot(feature) < p2.second.get_means().dot(feature);
                            })->first;
  }

};

#endif //MOCHIMOCHI_NHERD_HPP_
