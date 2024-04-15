#pragma once
#include "param.cuh"
#include "ops/op_adam.cuh"

template<typename T>
class ADAM {
    std::vector<Parameter<T> *> params;
    float lr;
    float beta_1;
    float beta_2;
    float epsilon;
    float lambda; // weight_decay
    // float vt_max = 0.0;
    float t = 0;

    public:
    std::vector<Tensor<T>*> mt;
    std::vector<Tensor<T>*> vt;
    ADAM(std::vector<Parameter<T>*> params_, float lr_, float beta_1_, float beta_2_, float epsilon_, float lambda): 
    params(params_), lr(lr_), beta_1(beta_1_), beta_2(beta_2_), epsilon(epsilon_), lambda(lambda)
    {
        std:: cout << "hyperparameters: " << std::endl;
        std:: cout << "\tlr: " << lr << ", beta_1: " << beta_1 << ", beta_2: " 
                   << beta_2 << ", eps: " << epsilon << ", lambda: " << lambda << std::endl;

    }
    
    void step() {
        if (t == 0) {
            for(auto pp: params) {
                Tensor<T>* m0 = new Tensor<T>{pp->t.h, pp->t.w, true};
                Tensor<T>* v0 = new Tensor<T>{pp->t.h, pp->t.w, true};
                op_const_init(*m0, 0.0);
                op_const_init(*v0, 0.0);
                mt.emplace_back(m0);
                vt.emplace_back(v0);
            }
        }
        t += 1;
        // std::cout << "iter: " << t << std::endl;


        for (int i = 0; i < params.size(); i++) {
            Parameter<T>* pp = params[i];
            Tensor<T> mm = *mt[i];
            Tensor<T> vv = *vt[i];

            op_adam(pp->t, pp->dt, pp->t, lr, beta_1, beta_2, epsilon, lambda, mm, vv, t);
        }
        // std::cout << mt[0]->str() << std::endl;
        // std::cout << vt[0]->str() << std::endl;

    } 
};
