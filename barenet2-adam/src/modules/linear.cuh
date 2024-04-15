#pragma once
#include "modules/param.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_mm.cuh"

template<typename T>
class LinearLayer {
    private:
        int in_dim;
        int out_dim;

        Parameter<T> w;
        Parameter<T> b;

    public:
    LinearLayer(int in_dim_, int out_dim_, bool gpu):in_dim(in_dim_), out_dim(out_dim_) {
        w = Parameter<T>{in_dim, out_dim, gpu};
        b = Parameter<T>{1, out_dim, gpu};
    }

    LinearLayer() {}
    
    LinearLayer(LinearLayer&& other) : in_dim(other.in_dim), out_dim(other.out_dim), w(other.w), b(other.b) {}
                
    std::vector<Parameter<T>*> parameters() {
        std::vector<Parameter<T> *> v;
        v.push_back(&w);
        v.push_back(&b);
        return v;
    }
    
    void init_uniform() {
        // Do Kaiming uniform
        float max = 1.0f / std::sqrt(in_dim);
        op_uniform_init(w.t, -max, max);
        op_uniform_init(b.t, -max, max);
        //std::cout << "init b=" << b.t.str() << std::endl;
    }

    //This function calculates the output of a lienar layer 
    //and stores the result in tensor "y"
    void forward(const Tensor<float> &x, Tensor<float> &y) {
        //Lab-2: please add your code here
        // y = x @ w + b
        // b in dim of 1 * out
        Tensor<float> tmp(y.h, y.w, true); // shit! cannot do implace operations!!!
        op_mm(x, w.t, tmp);
        // op_mm(w.t.transpose(), x.transpose(), tmp);
        // assert(op_allclose(tmp.transpose(), y)); // test transpose
        op_add(tmp, b.t, y);
    }

    //This function performs the backward operation of a linear layer
    //Suppose y = Linear(x). Then function argument "dy" is the gradients of "y", 
    //and function argument "x" is the saved x.
    //This function compute the weight gradients (dw, db) and saves them in w.dt and b.dt respectively
    //It also computes the graidents of "x" and saves it in dx.
    void backward(const Tensor<float> &x, const Tensor<float> &dy, Tensor<float> &dx) {
        //Lab-2: Please add your code here
        //dw = dy * dy/dw = x^T @ dy nxm = nxb @ bxm
        //db = dy * dy/db = 1^T @ dy 1xm = 1xb @ bxm
        //dx = dy * dy/dx = dy @ w^T bxn = bxm @ mxn

        op_mm(x.transpose(), dy, w.dt);
        Tensor<float> ones{1, x.h, true};
        op_const_init(ones, 1.0); 
        op_mm(ones, dy, b.dt);
        op_mm(dy, w.t.transpose(), dx);

        // std::cout << "x: " << Index(x.toHost(), 0, 0) << std::endl;
        // std::cout << " dy: " << Index(dy.toHost(), 0, 0) << std::endl;
        // std::cout << " w.dt: " << Index(w.dt.toHost(), 0, 0) << std::endl;
        // std::cout << " b.dt: " << Index(b.dt.toHost(), 0, 0) << std::endl;
        // std::cout << " dx: " << Index(dx.toHost(), 0, 0) << std::endl;
                    
    }
};
