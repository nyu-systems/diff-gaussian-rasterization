#pragma once
#include "utils/tensor.cuh"
#include "ops/op_elemwise.cuh"

template <typename T>
class Parameter{
    public:
    Parameter(int t_h, int t_w, bool gpu) {
        t = Tensor<T>{t_h, t_w, gpu};
        dt = Tensor<T>{t_h, t_w, gpu};
    }
    Parameter() {}
    Tensor<T> t;
    Tensor<T> dt;
};