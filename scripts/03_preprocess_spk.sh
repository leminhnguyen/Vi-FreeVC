source /home/nguyenlm/anaconda3/etc/profile.d/conda.sh
conda activate free-vc

nice CUDA_VISIBLE_DEVICES=1 python preprocess_spk.py \
    --in_dir dataset/DUMMY_16k/vi \
    --out_dir_root dataset \
    --num_workers 20