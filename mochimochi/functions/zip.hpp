#ifndef MOCHIMOCHI_FUNCTIONS_ZIP_HPP_
#define MOCHIMOCHI_FUNCTIONS_ZIP_HPP_

#include <boost/iterator/zip_iterator.hpp>
#include <boost/range.hpp>

namespace functions {
  template <typename... T>
  auto zip(const T&... containers) -> boost::iterator_range<boost::zip_iterator<decltype(boost::make_tuple(std::begin(containers)...))>> {
    auto zip_begin = boost::make_zip_iterator(boost::make_tuple(std::begin(containers)...));
    auto zip_end = boost::make_zip_iterator(boost::make_tuple(std::end(containers)...));
    return boost::make_iterator_range(zip_begin, zip_end);
  }
};

#endif //MOCHIMOCHI_FUNCTIONS_ZIP_HPP_
