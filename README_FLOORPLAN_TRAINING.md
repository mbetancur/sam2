# SAM2 Floor Plan Fine-tuning Guide

This guide walks you through fine-tuning SAM2 on your floor plan dataset to improve room and hall segmentation.

## Overview

This fine-tuning process will:
- Improve SAM2's ability to segment rooms and halls in floor plans
- Enhance both prompt-based and automatic segmentation
- Adapt the model to architectural/floor plan visual patterns

## Prerequisites

- **Hardware**: GPU with at least 80GB memory (A100 recommended)
- **Dataset**: 30 floor plan images with palettised PNG masks
- **Data Structure**: Already organized correctly in `tuning_dataset/`:
  ```
  tuning_dataset/
  ├── images/
  │   ├── floorplan1/
  │   │   └── 00000.png
  │   ├── floorplan2/
  │   │   └── 00000.png
  │   └── ... (up to floorplan30)
  └── annotations/
      ├── floorplan1/
      │   └── 00000.png  (palettised mask)
      ├── floorplan2/
      │   └── 00000.png
      └── ... (up to floorplan30)
  ```

## Step-by-Step Process

### 1. Validate Your Dataset (Optional)

Before training, validate that your dataset is correctly formatted:

```bash
python scripts/validate_floorplan_data.py ./tuning_dataset
```

This will:
- Verify all 30 folders exist
- Check that masks are palettised PNG format
- Verify image/mask alignment
- Report statistics about instances per floor plan

### 2. Create Train/Validation Split

Split your 30 floor plans into training (24) and validation (6) sets:

```bash
python scripts/create_train_val_split.py ./tuning_dataset --seed 42
```

This creates:
- `tuning_dataset/train_list.txt` - List of training folder names
- `tuning_dataset/val_list.txt` - List of validation folder names

### 3. Update Training Configuration

Edit `sam2/configs/sam2.1_training/sam2.1_hiera_b+_floorplan_finetune.yaml` and update:

```yaml
dataset:
  img_folder: ./tuning_dataset/images  # Path to your images folder
  gt_folder: ./tuning_dataset/annotations  # Path to your annotations folder
  file_list_txt: ./tuning_dataset/train_list.txt  # Path to train list
```

**Key Configuration Parameters:**
- `train_batch_size`: 2 (can increase to 3-4 for 80GB GPU)
- `num_frames`: 1 (single frame images)
- `max_num_objects`: 15 (adjust based on your data - check validation output)
- `num_epochs`: 40 (may need more with only 30 images)
- `gpus_per_node`: 1 (for Colab/single GPU)

### 4. Download SAM2.1 Checkpoint

Download the base model checkpoint:

```bash
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
cd ..
```

### 5. Install Dependencies

Install SAM2 with training dependencies:

```bash
pip install -e ".[dev]"
```

### 6. Start Training

For Colab or single GPU:

```bash
python training/train.py \
    -c sam2/configs/sam2.1_training/sam2.1_hiera_b+_floorplan_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 1
```

For multi-GPU (if available):

```bash
python training/train.py \
    -c sam2/configs/sam2.1_training/sam2.1_hiera_b+_floorplan_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 8
```

### 7. Monitor Training

Training logs and TensorBoard files are saved to:
```
sam2_logs/sam2.1_hiera_b+_floorplan_finetune/
├── tensorboard/  # TensorBoard logs
├── logs/         # Text logs
└── checkpoints/  # Model checkpoints
```

To view TensorBoard:
```bash
tensorboard --logdir sam2_logs/sam2.1_hiera_b+_floorplan_finetune/tensorboard
```

### 8. Using the Fine-tuned Model

After training, use your fine-tuned checkpoint just like the original SAM2:

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load your fine-tuned checkpoint
checkpoint = "./sam2_logs/sam2.1_hiera_b+_floorplan_finetune/checkpoints/checkpoint_epoch_40.pt"
model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Use for inference
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(your_floor_plan_image)
    masks, scores, logits = predictor.predict(point_coords=[[x, y]], point_labels=[1])
```

## Using Google Colab

For Colab training, use the provided notebook:
- Open `notebooks/floorplan_training_colab.ipynb` in Colab
- Follow the step-by-step cells
- Make sure to select A100 GPU (80GB) in Colab settings

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `train_batch_size` in the config (try 1 instead of 2)
- Reduce `max_num_objects` if you have many rooms per floor plan
- Reduce `num_train_workers`

### Slow Training
- Increase `train_batch_size` if you have memory headroom
- Reduce `num_epochs` if you see convergence early
- Check GPU utilization with `nvidia-smi`

### Poor Results
- Increase `num_epochs` (with 30 images, may need 50-60 epochs)
- Check that masks are properly palettised
- Verify `max_num_objects` matches your data (use validation script output)
- Consider data augmentation (already enabled in config)

### Image Format
- **PNG is preferred** for floor plans (lossless, better for line drawings and text)
- The code now supports both PNG and JPG formats
- PNG images will be used automatically if available
- No conversion needed - PNG works perfectly for this use case

## Expected Training Time

- **A100 80GB GPU**: ~2-4 hours for 40 epochs
- **V100 32GB GPU**: ~4-6 hours (may need batch_size=1)
- **T4 16GB GPU**: Not recommended (insufficient memory)

## Results

After fine-tuning, you should see:
- Better segmentation accuracy on floor plans
- Improved recognition of rooms, halls, and architectural elements
- Better handling of floor plan-specific visual patterns
- Improved performance on both prompt-based and automatic segmentation

## Next Steps

1. Evaluate on your validation set
2. Test on new floor plan images
3. Adjust hyperparameters if needed
4. Consider collecting more data if results are not satisfactory

## Files Created

- `scripts/validate_floorplan_data.py` - Dataset validation script
- `scripts/create_train_val_split.py` - Train/val split creation
- `sam2/configs/sam2.1_training/sam2.1_hiera_b+_floorplan_finetune.yaml` - Training configuration
- `notebooks/floorplan_training_colab.ipynb` - Colab training notebook
- `README_FLOORPLAN_TRAINING.md` - This guide

## References

- [SAM2 Training README](training/README.md)
- [SAM2 Main README](README.md)
- [SAM2 Paper](https://arxiv.org/abs/2408.00714)

