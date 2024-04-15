#pragma once
#include "modules/linear.cuh"
template <typename T>
class MLP
{
private:
    std::vector<LinearLayer<T>> layers;
    std::vector<int> layer_dims;
    std::vector<Tensor<T>> activ;
    std::vector<Tensor<T>> d_activ;

    int batch_size;
    int in_dim;

public:
    MLP(int batch_size_, int in_dim_, std::vector<int> layer_dims_, bool gpu)
        : batch_size(batch_size_), in_dim(in_dim_), layer_dims(layer_dims_)
    {
        assert(layer_dims.size() > 1);
        for (int i = 0; i < layer_dims.size(); i++)
        {
            if (i == 0)
            {
                layers.emplace_back(in_dim, layer_dims[i], gpu);
            }
            else
            {
                layers.emplace_back(layer_dims[i - 1], layer_dims[i], gpu);
            }
        }
        // make all the activation tensors
        activ.reserve(layer_dims.size() - 1);
        d_activ.reserve(layer_dims.size() - 1);
        for (int i = 0; i < layer_dims.size() - 1; i++)
        {
            activ.emplace_back(Tensor<T>(batch_size, layer_dims[i], gpu));
            // technically, i do not need to save d_activ for backprop, but since iterative
            // training does repeated backprops, reserving space for these tensor once is a good idea
            d_activ.emplace_back(Tensor<T>(batch_size, layer_dims[i], gpu));
        }
    }

    std::vector<Parameter<T> *> parameters()
    {
        std::vector<Parameter<T> *> params;
        for (int i = 0; i < layer_dims.size(); i++)
        {
            auto y = layers[i].parameters();
            params.insert(params.end(), y.begin(), y.end());
        }
        return params;
    }

    void init()
    {
        for (int i = 0; i < layers.size(); i++)
        {
            layers[i].init_uniform();
        }
    }

    // This function peforms the forward operation of a MLP model
    // Specifically, it should call the forward oepration of each linear layer
    // Except for the last layer, it should invoke Relu activation after each layer.
    void forward(const Tensor<T> &in, Tensor<T> &out)
    {
        // Lab-2: add your code here
        Tensor<T> relu_out;

        layers[0].forward(in, activ[0]);
        relu_out = Tensor<T>{activ[0].h, activ[0].w, true};
        op_relu(activ[0], relu_out); // TODO: TBD: I think in this lab, we always use relu

        for (int i = 1; i <= layers.size() - 2; i++)
        {
            layers[i].forward(relu_out, activ[i]);
            relu_out = Tensor<T>{activ[i].h, activ[i].w, true};
            op_relu(activ[i], relu_out);
        }
        layers[layers.size() - 1].forward(relu_out, out); 
    }

    // This function perofmrs the backward operation of a MLP model.
    // Tensor "d_out" is the gradients for the outputs of the last linear layer (aka d_logits from op_cross_entropy_loss)
    // Invoke the backward function of each linear layer and Relu from the last one to the first one.
    void backward(const Tensor<T> &in, const Tensor<T> &d_out, Tensor<T> &d_in)
    {
        // in: b_images             -  x,
        // d_out: d_logits          -  dy,
        // d_in: d_input_images     -  dx
        // Lab-2: add your code here
        // std::cout << "#layers: " << layers.size() << std::endl;
        // std::cout << "layers: " << layers.size()-1 << std::endl;
        Tensor<T> relu_out;
        relu_out = Tensor<T>{activ[layers.size() - 2].h, activ[layers.size() - 2].w, true};
        op_relu(activ[layers.size() - 2], relu_out);
        layers[layers.size() - 1].backward(relu_out, d_out, d_activ[layers.size() - 2]);
        for (int i = layers.size() - 2; i >= 1; i--)
        {
            // Tensor<float> tmp = {d_activ[i].h, d_activ[i].w, true};

            op_relu_back(activ[i], d_activ[i], d_activ[i]); // relu_back(x, dy)

            relu_out = Tensor<T>{activ[i-1].h, activ[i-1].w, true};
            op_relu(activ[i-1], relu_out);
            layers[i].backward(relu_out, d_activ[i], d_activ[i - 1]);

            // std::cout << "layers: " << i << std::endl;
            // std::cout << "dactiv: " << d_activ[i].str() << std::endl;
        }
        // Tensor<float> tmp0 = {d_activ[0].h, d_activ[0].w, true};
        op_relu_back(activ[0], d_activ[0], d_activ[0]); // relu_back(x, dy)
        layers[0].backward(in, d_activ[0], d_in);
    }
};
