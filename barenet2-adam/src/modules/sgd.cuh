#pragma once
#include "param.cuh"

template<typename T>
class SGD {
    std::vector<Parameter<T> *> params;
    float lr;
    public:
    SGD(std::vector<Parameter<T>*> params_, float lr_): params(params_), lr(lr_) {}
    void step() {
        for(auto pp: params) {
            op_sgd(pp->t, pp->dt, pp->t, lr);
        }
    } 
};
