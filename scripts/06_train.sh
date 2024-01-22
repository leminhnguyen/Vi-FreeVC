source /home/nguyenlm/miniconda3/etc/profile.d/conda.sh
conda activate freevc

python train.py \
    --config configs/freevc-nosr.json \
    --model vi