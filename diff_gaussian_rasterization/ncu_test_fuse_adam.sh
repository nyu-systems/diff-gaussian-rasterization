#!/bin/bash

# Check if the user has provided the required number of inputs
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <test_name> <command_number>"
    echo "Command Numbers: 0 for all, 1 for single, 2 for multi, 3 for PyTorch"
    exit 1
fi

test_name=$1
command_number=$2

# Ensure the logs directory exists
mkdir -p logs

# Select the command to run based on the second parameter
case $command_number in
    0)
        # Run all profiling on three type kernel
        ncu --metrics gpu__time_duration.sum,dram__bytes.sum --kernel-name op_adam_single_tensor_kernel --target-processes all python diff_gaussian_rasterization/test_fuse_adam.py --optimizer fused_single > logs/single-${test_name}.log

        ncu --metrics gpu__time_duration.sum,dram__bytes.sum --kernel-name op_adam_multi_tensor_kernel --target-processes all python diff_gaussian_rasterization/test_fuse_adam.py --optimizer fused_multi > logs/multi-${test_name}.log

        ncu --metrics gpu__time_duration.sum,dram__bytes.sum --kernel-name multi_tensor_apply_kernel --target-processes all python diff_gaussian_rasterization/test_fuse_adam.py --optimizer torch_adam > logs/pytorch-${test_name}.log
        ;;
    1)
        # Run profiling on single tensor kernel
        ncu --metrics gpu__time_duration.sum,dram__bytes.sum --kernel-name op_adam_single_tensor_kernel --target-processes all python diff_gaussian_rasterization/test_fuse_adam.py --optimizer fused_single > logs/single-${test_name}.log
        ;;
    2)
        # Run profiling on multi tensor kernel
        ncu --metrics gpu__time_duration.sum,dram__bytes.sum --kernel-name op_adam_multi_tensor_kernel --target-processes all python diff_gaussian_rasterization/test_fuse_adam.py --optimizer fused_multi > logs/multi-${test_name}.log
        ;;
    3)
        # Run profiling on PyTorch multi tensor apply kernel
        ncu --metrics gpu__time_duration.sum,dram__bytes.sum --kernel-name multi_tensor_apply_kernel --target-processes all python diff_gaussian_rasterization/test_fuse_adam.py --optimizer torch_adam > logs/pytorch-${test_name}.log
        ;;
    *)
        echo "Invalid command number. Please use 0 for all, 1 for single, 2 for multi, or 3 for PyTorch."
        exit 1
        ;;
esac