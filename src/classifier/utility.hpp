#ifndef MOCHIMOCHI_UTILITY_HPP_
#define MOCHIMOCHI_UTILITY_HPP_

#include <vector>

namespace utility {
  template <typename IteratorT, typename FunctionT>
  FunctionT enumerate(IteratorT begin,
                      IteratorT end,
                      typename std::iterator_traits<IteratorT>::difference_type initial,
                      FunctionT func) {

    for (; begin != end; ++begin, ++initial) {
      func(initial, *begin);
    }

    return func;
  }
};

#endif //MOCHIMOCHI_UTILITY_HPP_
