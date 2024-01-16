source /home/nguyenlm/anaconda3/etc/profile.d/conda.sh
conda activate free-vc

python preprocess_flist.py \
    --train_list filelists/train-vi.txt \
    --test_list filelists/test-vi.txt \
    --val_list filelists/val-vi.txt \
    --source_dir dataset/DUMMY/vi