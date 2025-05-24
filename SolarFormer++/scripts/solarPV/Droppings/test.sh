# Set GPU
export CUDA_VISIBLE_DEVICES=1

# Define base path
PathToSolarFormer= your_path_to_SolarFormer++

# Define individual paths
model_path="${PathToSolarFormer}/Pre-Trained_Models/Droppings/"
img_path="${PathToSolarFormer}/mask2former/SolarPV/Droppings/test_imgs"
output_path="${PathToSolarFormer}/demo_outputs/Droppings"

# Create output directory if it doesn't exist
mkdir -p "${output_path}"

# Run inference
python "${PathToSolarFormer}/demo/demo.py" --config-file "${model_path}config.yaml" \
  --input "${img_path}/*.png" \
  --output "${output_path}" \
  --confidence-threshold 0.95 \
  --opts MODEL.WEIGHTS "${model_path}model_final.pth"
