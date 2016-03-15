#ifndef MOCHIMOCHI_FUNCTIONS_HPP_
#define MOCHIMOCHI_FUNCTIONS_HPP_

#include <vector>

namespace functions {
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

#endif //MOCHIMOCHI_FUNCTIONS_HPP_
