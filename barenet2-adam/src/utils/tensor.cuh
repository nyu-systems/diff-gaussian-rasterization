#pragma once
#include <assert.h>
#include <random>
#include <memory>
#include <sstream>
#include <string>

#include "utils/assert.cuh"

#define ISCLOSE_RELTOL 1e-6 // this would not work for precision lower than float
#define ISCLOSE_ABSTOL 1e-6

// Index is a MACRO that returns the element of t tensor at row, col coordinate
#define Index(t, row, col) ((((t).rawp)[(t).offset + (row) * (t).stride_h + (col) * (t).stride_w]))
// IndexOutofBound is a MACRO to test whether coordinate row,col is considered out of bounds for tensor "t"
#define IndexOutofBound(t, row, col) ((((row) >= (t).h) || ((col) >= (t).w)))

template <typename T>
struct cudaDeleter
{
  void operator()(T *p) const
  {
    if (p != nullptr)
    {
      // std::cout << "Free p=" << p << std::endl;
      cudaFree(p);
    }
  }
};

template <typename T>
struct cpuDeleter
{
  void operator()(T *p) const
  {
    if (p != nullptr)
    {
      // std::cout << "Free p=" << p << std::endl;
      free(p);
    }
  }
};

template <typename T>
class Tensor
{
public:
  int32_t h; // height
  int32_t w; // width
  int32_t stride_h;
  int32_t stride_w;
  int32_t offset;
  T *rawp;
  std::shared_ptr<T> ref; // refcounted pointer, for garbage collection use only
  bool on_device;

  Tensor() : h(0), w(0), stride_h(0), stride_w(0), offset(0), rawp(nullptr), on_device(false)
  {
    ref = std::shared_ptr<T>(rawp, cpuDeleter<T>());
  }

  Tensor(int32_t h_, int32_t w_, bool on_device_ = false)
      : h(h_), w(w_), stride_h(w_), stride_w(1), offset(0), on_device(on_device_)
  {
    if (on_device_)
    {
      cudaAssert(cudaMalloc(&rawp, sizeof(T) * h * w));
      // std::cout << "cudaMalloc p=" << rawp << std::endl;
      ref = std::shared_ptr<T>(rawp, cudaDeleter<T>());
    }
    else
    {
      rawp = (T *)malloc(sizeof(T) * h * w);
      // std::cout << "malloc p=" << rawp << " h=" << h << " w=" << w << std::endl;
      ref = std::shared_ptr<T>(rawp, cpuDeleter<T>());
    }
  }

  void toHost(Tensor<T> &out) const
  {
    assert(!out.on_device);
    assert(h == out.h && w == out.w);
    if (!on_device)
    {
      out = *this;
      return;
    }
    out.offset = offset;
    out.stride_h = stride_h;
    out.stride_w = stride_w;
    cudaAssert(cudaMemcpy(out.rawp, rawp, h * w * sizeof(T), cudaMemcpyDeviceToHost));
  }

  Tensor<T> toHost() const
  {
    Tensor<T> t{h, w};
    toHost(t);
    return t;
  }

  void toDevice(Tensor<T> &out) const
  {
    assert(out.on_device);
    assert(h == out.h && w == out.w);
    if (on_device)
    {
      out = *this;
      return;
    }
    out.offset = offset;
    out.stride_h = stride_h;
    out.stride_w = stride_w;
    cudaAssert(cudaMemcpy(out.rawp, rawp, h * w * sizeof(T), cudaMemcpyHostToDevice));
  }

  Tensor<T> toDevice() const
  {
    Tensor<T> t{h, w, true};
    toDevice(t);
    return t;
  }

  Tensor<T> transpose() const
  {
    Tensor<T> t{};
    t.w = h;
    t.stride_w = stride_h;
    t.h = w;
    t.stride_h = stride_w;
    t.offset = offset;
    t.ref = ref;
    t.rawp = rawp;
    t.on_device = on_device;
    return t;
  }

  Tensor<T> slice(int start_h, int end_h, int start_w, int end_w) const
  {
    Tensor<T> t{};
    assert(start_h < end_h && end_h <= h);
    assert(start_w < end_w && end_w <= w);
    t.w = end_w - start_w;
    t.h = end_h - start_h;
    t.stride_h = stride_h;
    t.stride_w = stride_w;
    t.ref = ref;
    t.rawp = rawp;
    t.offset = offset + start_h * stride_h + start_w * stride_w;
    t.on_device = on_device;
    return t;
  }

  std::string str() const
  {
    Tensor<T> t{};
    if (on_device)
    {
      t = toHost();
    }
    else
    {
      t = *this;
    }
    std::stringstream ss;
    for (int i = 0; i < h; i++)
    {
      for (int j = 0; j < w; j++)
      {
        if (std::is_same_v<T, char> || std::is_same_v<T, unsigned char>)
        {
          ss << (int)Index(t, i, j) << " ";
        }
        else
        {
          // std::cout << "haha " << Index(t, i, j) << std::endl;
          ss << Index(t, i, j) << " ";
        }
        ss << "";
      }
      ss << "\n";
    }
    return ss.str();
  }

  T mean() const
  {
    assert(!on_device);
    T sum = 0;
    for (int i = 0; i < h; i++)
    {
      for (int j = 0; j < w; j++)
      {
        sum += Index(*this, i, j);
      }
    }
    return sum / (h * w);
  }

  T range() const
  {
    assert(!on_device);
    T min, max;
    min = max = Index(*this, 0, 0);
    for (int i = 0; i < h; i++)
    {
      for (int j = 0; j < w; j++)
      {
        T e = Index(*this, i, j);
        if (e > max)
        {
          max = e;
        }
        else if (e < min)
        {
          min = e;
        }
      }
    }
    std::cout << max << " " << min << std::endl;
    return max - min;
  }

  T max() const
  {
    assert(!on_device);
    T min, max;
    min = max = Index(*this, 0, 0);
    for (int i = 0; i < h; i++)
    {
      for (int j = 0; j < w; j++)
      {
        T e = Index(*this, i, j);
        if (e > max)
        {
          max = e;
        }
        else if (e < min)
        {
          min = e;
        }
      }
    }
    return max;
  }
};