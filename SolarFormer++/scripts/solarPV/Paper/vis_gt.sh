# Define base path
PathToSolarFormer=your_path_to_SolarFormer++

# Set environment variables
export DETECTRON2_DATASETS="${PathToSolarFormer}/mask2former"

# Define paths
config_file="${PathToSolarFormer}/configs/solarPV/Paper/Ign_Base-Cityscapes-SemanticSegmentation.yaml"
output_dir="${PathToSolarFormer}/Media/Paper/Ign_Base-Cityscapes-SemanticSegmentation"

# Create output directory if it doesn't exist
mkdir -p "${output_dir}"

# Run visualization
python "${PathToSolarFormer}/visualize_data.py" --config-file "${config_file}" \
    --output-dir "${output_dir}" \
    --source "annotation" \
    --split "train"