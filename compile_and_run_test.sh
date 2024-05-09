cd ../..
pip install -e submodules/diff-gaussian-rasterization
cd submodules/diff-gaussian-rasterization
python diff_gaussian_rasterization/test_fuse_adam.py --optimizer fused_multi