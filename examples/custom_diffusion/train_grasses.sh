
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="model"
export INSTANCE_DIR="./data/grasses"

accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
#  --class_data_dir=./real_reg/samples_cat/ \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --class_prompt="grasses" \
#  --num_class_images=200 \
  --instance_prompt="photo of a grasses"  \
  --resolution=1024  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --scale_lr --hflip  \
  --modifier_token "<grasses>"

