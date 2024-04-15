/* This file is based on pytorch's mnist dataloader
pytorch/torch/csrc/api/src/data/datasets/mnist.cpp
with minor changes*/

#pragma once

#include "utils/tensor.cuh"
#include <fstream>

class MNIST {
    public:  
    //the mode in which the dataset is loaded
    enum class Mode { kTrain, kTest };

    Tensor<float> images;
    Tensor<char> targets;

    static const int kTrainSize = 60000;
    static const int kTestSize = 10000;
    static const int kImageMagicNumber = 2051;
    static const int kTargetMagicNumber = 2049;
    static const int kImageRows = 28;
    static const int kImageColumns = 28;
    static constexpr std::string_view kTrainImagesFilename = "train-images-idx3-ubyte";
    static constexpr std::string_view kTrainTargetsFilename = "train-labels-idx1-ubyte";
    static constexpr std::string_view kTestImagesFilename = "t10k-images-idx3-ubyte";
    static constexpr std::string_view kTestTargetsFilename = "t10k-labels-idx1-ubyte";

    MNIST(const std::string& root, Mode mode = Mode::kTrain) 
    : images(read_images(root, mode==Mode::kTrain)), targets(read_targets(root, mode==Mode::kTrain)) {
        //only runs on a little endian machine, assert endianness
        unsigned int x = 0x76543210;
        char *c = (char*) &x;
        assert(*c == 0x10);
    }

    private:

    Tensor<float> read_images(const std::string &root, bool train) {
        std::string path = root + "/" + (train? std::string(kTrainImagesFilename) : std::string(kTestImagesFilename));
        std::cout << path << std::endl;
        std::ifstream images_stream(path, std::ios::binary);
        const auto count = train ? MNIST::kTrainSize : MNIST::kTestSize;

        // From http://yann.lecun.com/exdb/mnist/
        expect_int32(images_stream, MNIST::kImageMagicNumber);
        expect_int32(images_stream, count);
        expect_int32(images_stream, MNIST::kImageRows);
        expect_int32(images_stream, MNIST::kImageColumns);

        Tensor<float> t{count, MNIST::kImageRows*MNIST::kImageColumns}; 
        for (int i = 0; i < count*MNIST::kImageRows*MNIST::kImageColumns; i++) {
            uint8_t val;
            images_stream.read(reinterpret_cast<char*>(&val), sizeof(val));
            change_endianness(reinterpret_cast<char*>(&val), sizeof(val));
            float v = (float)val / 255.0;
            assert(v>=0);
            t.rawp[i] = (float)val / 255.0;
        }
        return t;
    }

    Tensor<char> read_targets(const std::string &root, bool train) {
        std::string path = root + "/" + (train ? std::string(kTrainTargetsFilename) : std::string(kTestTargetsFilename));
        std::ifstream targets_stream(path, std::ios::binary);
        const auto count = train ? MNIST::kTrainSize : MNIST::kTestSize;

        expect_int32(targets_stream, MNIST::kTargetMagicNumber);
        expect_int32(targets_stream, count);

        Tensor<char> t{count, 1}; 
        targets_stream.read(reinterpret_cast<char*>(t.rawp), count);
        return t;      
    }

    void change_endianness(char *buf, int sz) {
        char tmp;
        for (int i = 0; i < sz/2; i++) {
            tmp = buf[i];
            buf[i] = buf[sz-1-i];
            buf[sz-1-i] = tmp;
        }
    }

    void expect_int32(std::ifstream& stream, uint32_t expected) {
        uint32_t value;
        stream.read(reinterpret_cast<char*>(&value), sizeof(value));
        change_endianness(reinterpret_cast<char*>(&value), sizeof(value));
        assert(value == expected);
    }

};