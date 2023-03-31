#!/bin/bash

RAW_DATA_DIR=<dir-to-dodh-data>
SAVE_DIR=<your-save-dir>
BATCH_SIZE=8

python train.py \
    --dataset 'icbeb' \
    --raw_data_dir $RAW_DATA_DIR \
    --num_nodes 12 \
    --input_dim 1 \
    --output_dim 9 \
    --train_batch_size $BATCH_SIZE \
    --test_batch_size $BATCH_SIZE \
    --num_workers 8 \
    --model_name 'graphs4mer' \
    --graph_learn_metric "self_attention" \
    --dropout 0.1 \
    --g_conv 'gine' \
    --num_gcn_layers 1 \
    --hidden_dim 128 \
    --num_temporal_layers 4 \
    --state_dim 64 \
    --bidirectional True \
    --temporal_model 's4' \
    --temporal_pool 'mean' \
    --gin_mlp True \
    --train_eps True \
    --graph_pool 'mean' \
    --activation_fn 'leaky_relu' \
    --prune_method 'thresh_abs' \
    --thresh 0.02 \
    --use_prior False \
    --knn 2 \
    --residual_weight 0.6 \
    --regularizations 'feature_smoothing' 'degree' 'sparse' \
    --feature_smoothing_weight 1.0 \
    --degree_weight 0.0 \
    --sparse_weight 0.5 \
    --save_dir $SAVE_DIR \
    --metric_name 'auroc' \
    --eval_metrics 'auroc' 'fbeta' 'gbeta' \
    --metric_avg 'macro' \
    --find_threshold_on 'F1' \
    --lr_init 1e-3 \
    --l2_wd 1e-3 \
    --num_epochs 100 \
    --scheduler timm_cosine \
    --t_initial 100 \
    --warmup_t 5 \
    --optimizer adamw \
    --patience 20 \
    --do_train True \
    --gpus 1
