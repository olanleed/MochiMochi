## USAGE

```
$ clang++ -std=c++11 adagrad_rda.cpp -lboost_program_options -o rda
$ ./rda --dim <dimension_size> --train <traindata_path> --test <testdata_path> --lambda 0.000001
```