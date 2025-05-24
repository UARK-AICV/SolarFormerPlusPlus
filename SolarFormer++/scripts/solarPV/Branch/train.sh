# Define base path
PathToSolarFormer= your_path_to_SolarFormer++

export CUDA_VISIBLE_DEVICE=0
export DETECTRON2_DATASETS=$PathToSolarFormer/mask2former/


# Run training
python -W ignore $PathToSolarFormer/train_net.py --num-gpus 1 \
    --config-file $PathToSolarFormer/configs/solarPV/Branch/ign_maskformer_v2_R50.yaml \
    OUTPUT_DIR $PathToSolarFormer/train_outputs/Branch \
    DATALOADER.NUM_WORKERS 6 \