# DeepLab: Deep Labelling for Semantic Image Segmentation

This code is revised from [deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab) code.

This code generates bonnet segmentation model.

# Preparing data

```
python ./build_ade20k_data.py  \
  --train_image_folder="/home/ubuntu/data/bonnet_dataset/seg_and_img/images/training/" \
  --train_image_label_folder="/home/ubuntu/data/bonnet_dataset/seg_and_img/annotations/training/" \
  --val_image_folder="/home/ubuntu/data/bonnet_dataset/seg_and_img/images/validation/" \
  --val_image_label_folder="/home/ubuntu/data/bonnet_dataset/seg_and_img/annotations/validation/" \
  --output_dir="/home/ubuntu/data/bonnet_dataset/tfrecord"
```

# Training

Execute at /home/ubuntu/git/models/research/.

```
nohup python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=5000 \
    --train_split="train" \
    --model_variant="mobilenet_v2" \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="513,513" \
    --train_batch_size=4 \
    --base_learning_rate=3e-5 \
    --dataset="ade20k" \
    --initialize_last_layer \
    --quantize_delay_step=0 \
    --tf_initial_checkpoint="/home/ubuntu/data/bonnet_dataset_20200331/result_20200330_2/model.ckpt-10000" \
    --train_logdir="/home/ubuntu/data/bonnet_dataset/result_20200513" \
    --dataset_dir="/home/ubuntu/data/bonnet_dataset/tfrecord" >out20200513.log &
```

# Visualizing Result

```
nohup python deeplab/vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="mobilenet_v2" \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size="900,900" \
    --dataset="ade20k" \
    --checkpoint_dir="/home/ubuntu/data/bonnet_dataset/result_20200513"\
    --vis_logdir="/home/ubuntu/data/bonnet_dataset/result_20200513_img"\
    --dataset_dir="/home/ubuntu/data/bonnet_dataset/tfrecord" >out20200513_img.log &
```

# Evaluation

```
python deeplab/eval.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="mobilenet_v2" \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size="900,900" \
    --dataset="ade20k" \
    --checkpoint_dir="/home/ubuntu/data/bonnet_dataset/result_20200513"\
    --eval_logdir="/home/ubuntu/data/bonnet_dataset/result_20200513_eval"\
    --dataset_dir="/home/ubuntu/data/bonnet_dataset/tfrecord"
```

```
tensorboard --logdir='/home/ubuntu/data/bonnet_dataset/result_20200513_eval/'
```

```
python deeplab/export_model.py \
    --checkpoint_path="/home/ubuntu/data/bonnet_dataset/result_20200513/model.ckpt-2001"\
    --export_path="/home/ubuntu/frozen_inference_graph.pb"\
    --num_classes=151 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --quantize_delay_step=0 \
```
