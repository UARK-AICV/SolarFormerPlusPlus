## De-Solar Dataset

## 📦 De-Solar Dataset

The **De-Solar Dataset** is a high-quality UAV-based dataset developed to support obstruction localization and performance assessment in solar PV systems. It contains over 3,500 manually labeled images captured at altitudes ranging from 15 to 50 feet, each annotated with polygon masks for common surface obstructions such as branches, dirt, leaves, bird droppings, and paper. In addition to the image data, the dataset includes timestamp-aligned voltage readings and environmental metadata, enabling detailed analysis of how specific obstructions affect panel performance.

The dataset is located in the `De-Solar Dataset` folder and is organized into the following components:

- **`Voltage_Data/`** – Contains Excel files with image paths, voltage readings, and environmental variables.
- **`Original/`** – Includes the original UAV images, their corresponding annotation JSON files (LabelMe format), and segmentation masks.
- **`Cropped_Folder/`** – Contains cropped images of solar panels extracted from the originals. These images are used as input for model training.
- **`Ground_Folder/`** – Contains ground images from the dataset
This structure supports a complete training and evaluation pipeline for obstruction-aware solar PV analysis.

The De-Solar Dataset can be downloaded [Here](https://uark.box.com/s/89l7w5g5geeuhg9578wsc7998pdogjlu)

## SolarFormer++

**SolarFormer++** is a multi-scale Transformer-based segmentation model designed for comprehensive solar panel analysis. It performs both global-scale profiling using satellite imagery and fine-grained obstruction localization using UAV imagery. The model architecture integrates a ResNet backbone, a multi-scale Transformer encoder, and a masked-attention Transformer decoder to enhance segmentation precision, especially for small and visually similar objects. Trained and benchmarked on multiple public satellite datasets and the De-Solar Dataset, SolarFormer++ consistently outperforms existing deep learning approaches, making it an effective solution for real-world solar PV monitoring, degradation mitigation, and performance optimization.

We recommend following the installation guidelines provided in the SolarFormer++ repository, which includes all required modifications and custom scripts built on top of [Mask2Former](https://github.com/facebookresearch/Mask2Former). 

Inside the `scripts/` directory, you will find ready-to-use shell scripts for training, testing, demoing, and visualizing results for each obstruction type. 

> **Note**: Before running any script, be sure to update the `PathToSolarFormer` variable in the script to match your local directory structure.

## 🧪 Running with MMSegmentation

To use the **SolarFormer++** models implemented under the MMSegmentation framework, we recommend installing [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) by following the official installation guide.

Once installed, copy the contents of the `mmsegmentation/` folder from this repository and place the files into their corresponding locations in your MMSegmentation environment. These files include custom dataset definitions and configurations for different obstruction types.

We recommend using MMSegmentation's **default Docker workflow** for training new models and testing pre-trained models. Refer to their documentation for Docker usage and command-line training scripts.

### 🔧 Dataset Integration

To enable the custom solar PV datasets, you need to modify the dataset registry in `mmseg/datasets/__init__.py`:

1. **Add the following imports:**

```python
from .solarPV import SolarPV
from .solarPV_Branch import SolarPV_Branch
from .solarPV_Leaf import SolarPV_Leaf
from .solarPV_Paper import SolarPV_Paper
from .solarPV_Dirt import SolarPV_Dirt
from .solarPV_Droppings import SolarPV_Droppings
from .solarPV_Multi_Model import SolarPV_Multi_Model
from .droppings_OnlyObs import Droppings_OnlyObs
1. **Add the following imports:**
# SolarFormerPlusPlus
