source /home/nguyenlm/miniconda3/etc/profile.d/conda.sh
conda activate freevc

python preprocess_ssl.py \
    --in_dir dataset/DUMMY_16k/vi \
    --out_dir dataset/wavlm