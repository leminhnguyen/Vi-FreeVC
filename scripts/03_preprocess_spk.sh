source /home/nguyenlm/miniconda3/etc/profile.d/conda.sh
conda activate freevc

CUDA_VISIBLE_DEVICES=0 python preprocess_spk.py \
    --in_dir dataset/DUMMY_16k/vi \
    --out_dir_root dataset \
    --num_workers 5