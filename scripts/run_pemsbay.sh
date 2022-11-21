#!/bin/bash

RAW_DATA_DIR=<dir-to-pemsbay-data>
SAVE_DIR=<your-save-dir>
BATCH_SIZE=32

python train.py \
    --dataset 'pems_bay' \
    --task 'regression' \
    --raw_data_dir $RAW_DATA_DIR \
    --num_nodes 325 \
    --max_seq_len 12 \
    --output_seq_len 12 \
    --resolution 12 \
    --input_dim 2 \
    --output_dim 1 \
    --train_batch_size $BATCH_SIZE \
    --test_batch_size $BATCH_SIZE \
    --num_workers 8 \
    --adj_mat_dir 'data/pems_bay_sensor_graph/adj_mx_bay.pkl' \
    --model_name 'graphs4mer' \
    --graph_learn_metric "adaptive" \
    --adj_embed_dim 16 \
    --dropout 0.1 \
    --g_conv 'gine' \
    --num_gcn_layers 2 \
    --hidden_dim 256 \
    --temporal_model 's4' \
    --num_temporal_layers 4 \
    --state_dim 64 \
    --bidirectional True \
    --prune_method 'thresh' \
    --edge_top_perc 0.02 \
    --knn 3 \
    --activation_fn 'leaky_relu' \
    --use_prior False \
    --residual_weight 0.5 \
    --regularizations 'feature_smoothing' 'degree' 'sparse' \
    --feature_smoothing_weight 0.1 \
    --degree_weight 0.2 \
    --sparse_weight 0.2 \
    --save_dir $SAVE_DIR \
    --metric_name 'mae' \
    --eval_metrics 'mae' 'rmse' 'mape' \
    --lr_init 1e-3 \
    --l2_wd 5e-3 \
    --num_epochs 100 \
    --scheduler timm_cosine \
    --t_initial 100 \
    --warmup_t 5 \
    --optimizer adamw \
    --do_train True \
    --accumulate_grad_batches 1 \
    --gpus 1
