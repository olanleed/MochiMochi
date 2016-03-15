#ifndef MOCHIMOCHI_MSCW_HPP_
#define MOCHIMOCHI_MSCW_HPP_

#include <algorithm>
#include <unordered_map>
#include <boost/range/irange.hpp>
#include "../binary/scw.hpp"

class MSCW {
private:
  const std::size_t kClass;

private:
  std::unordered_map<std::size_t, SCW> _scws;

public:
  MSCW(const std::size_t dim, const std::size_t n_class, const double c, const double eta)
    : kClass(n_class) {
    static_assert(std::numeric_limits<decltype(n_class)>::max() > 2, "Class range Error. (n_class > 2)");

    for (const auto i : boost::irange<std::size_t>(1, kClass + 1)) {
      _scws.insert(std::pair<std::size_t, SCW>(i, SCW(dim, c, eta)) );
    }
  }

  virtual ~MSCW() { }

public:
  void update(const Eigen::VectorXd& feature, const std::size_t label) {
    for(auto& scw : _scws) {
      const auto t = (scw.first == label) ? 1 : -1;
      scw.second.update(feature, t);
    }
  }

  std::size_t predict(const Eigen::VectorXd& feature) const {
    return std::max_element(_scws.begin(), _scws.end(),
                            [&](const auto& p1, const auto& p2) {
                              return p1.second.get_means().dot(feature) < p2.second.get_means().dot(feature);
                            })->first;
  }

};

#endif //MOCHIMOCHI_MSCW_HPP_
