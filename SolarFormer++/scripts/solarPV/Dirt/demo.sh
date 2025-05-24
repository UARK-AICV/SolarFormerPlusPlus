# Define base path
PathToSolarFormer= your_path_to_SolarFormer++
model_output="${PathToSolarFormer}/train_outputs/Dirt/"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export DETECTRON2_DATASETS="${PathToSolarFormer}/mask2former"

# Run evaluation
python -W ignore "${PathToSolarFormer}/train_net.py" \
  --num-gpus 1 \
  --config-file "${model_output}config.yaml" \
  --eval-only \
  MODEL.WEIGHTS "${model_output}model_final.pth" \
  DATALOADER.NUM_WORKERS 4