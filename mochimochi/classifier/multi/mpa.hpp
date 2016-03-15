#ifndef MOCHIMOCHI_MPA_HPP_
#define MOCHIMOCHI_MPA_HPP_

#include <algorithm>
#include <unordered_map>
#include <boost/range/irange.hpp>
#include "../binary/pa.hpp"

class MPA {
private:
  const std::size_t kClass;

private:
  std::unordered_map<std::size_t, PA> _pas;

public:
  MPA(const std::size_t dim, const std::size_t n_class, const double C, const int select = 2)
    : kClass(n_class) {
    static_assert(std::numeric_limits<decltype(n_class)>::max() > 2, "Class range Error. (n_class > 2)");

    for (const auto i : boost::irange<std::size_t>(1, kClass + 1)) {
      _pas.insert(std::pair<std::size_t, PA>(i, PA(dim, C, select)) );
    }
  }

  virtual ~MPA() { }

public:
  void update(const Eigen::VectorXd& feature, const std::size_t label) {
    for(auto& pa : _pas) {
      const auto t = (pa.first == label) ? 1 : -1;
      pa.second.update(feature, t);
    }
  }

  std::size_t predict(const Eigen::VectorXd& feature) const {
    return std::max_element(_pas.begin(), _pas.end(),
                            [&](const auto& p1, const auto& p2) {
                              return p1.second.get_weight().dot(feature) < p2.second.get_weight().dot(feature);
                            })->first;
  }

};

#endif //MOCHIMOCHI_MPA_HPP_
