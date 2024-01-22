source /home/nguyenlm/miniconda3/etc/profile.d/conda.sh
conda activate freevc

python downsample.py \
    --in_dir dataset/DUMMY/vi \
    --out_dir1 dataset/DUMMY_16k/vi \
    --out_dir2 dataset/DUMMY_22k/vi