import argparse


def str2bool(string):
    if string == "true" or string == "True":
        return True
    else:
        return False


def get_args():
    parser = argparse.ArgumentParser(
        "Train GraphS4mer on multivariate signals."
    )

    # General args
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the outputs and checkpoints.",
    )
    parser.add_argument(
        "--save_output",
        type=str2bool,
        default="true",
        help="Whether to save model outputs.",
    )
    parser.add_argument(
        "--save_attn_weights",
        type=str2bool,
        default="false",
        help="Whether to save model outputs.",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Model checkpoint to start training/testing from.",
    )
    parser.add_argument(
        "--do_train", default=True, type=str2bool, help="Whether perform training."
    )
    parser.add_argument(
        "--freeze_s4", 
        default=False, 
        type=str2bool, 
        help="Whether to freeze pretrained S4."
    )
    parser.add_argument(
        "--s4_pretrained_dir",
        type=str,
        default=None,
        help="Dir to pretrained S4 model."
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs for training."
    )
    parser.add_argument(
        "--gpu_id", default=[0], type=int, nargs="+", help="List of GPU IDs."
    )

    ## Input args
    parser.add_argument(
        "--dataset",
        type=str,
        default="tuh",
        choices=(
            "tuh",
            "pems_bay",
            "dodh",
            "icbeb",
        ),
    )
    parser.add_argument(
        "--raw_data_dir", type=str, default=None, help="Dir to raw data."
    )
    parser.add_argument(
        "--adj_mat_dir",
        type=str,
        default=None,
        help="Dir to prior knowledge-based adj mat.",
    )
    parser.add_argument(
        "--preproc_dir",
        type=str,
        default=None,
        help="Dir to preprocessed freatures.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--output_seq_len", type=int, default=1, help="Output sequence length."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forecasting horizon. Only for forecasting tasks.",
    )
    parser.add_argument(
        "--sampling_freq",
        type=int,
        default=100,
        help="Sampling frequency for the dataset.",
    )
    parser.add_argument("--rand_seed", type=int, default=123, help="Random seed.")

    ## Model args
    parser.add_argument(
        "--model_name",
        type=str,
        default="graphs4mer",
        choices=(
            "graphs4mer",
            "s4",
            "temporal_gnn",
            "lstm",
        ),
        help="Name of model.",
    )
    ### General model args
    parser.add_argument(
        "--num_nodes", type=int, default=19, help="Number of nodes in graph."
    )
    parser.add_argument(
        "--num_gcn_layers", type=int, default=1, help="Number of graph conv layers."
    )
    parser.add_argument(
        "--num_temporal_layers",
        type=int,
        default=4,
        help="Number of temporal layers.",
    )
    parser.add_argument(
        "--input_dim", type=int, default=1, help="Input seq feature dim."
    )
    parser.add_argument(
        "--output_dim", type=int, default=1, help="Output seq feature dim."
    )
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument(
        "--graph_pool",
        type=str,
        default=None,
        choices=("max", "mean", "sum", None),
        help="Graph pooling operation.",
    )
    parser.add_argument(
        "--g_conv",
        type=str,
        default="gine",
        choices=("graphsage", "gat", "gine", "gcn"),
        help="Name of graph conv layer.",
    )
    parser.add_argument(
        "--activation_fn",
        type=str,
        default="leaky_relu",
        choices=("relu", "elu", "leaky_relu", "gelu"),
        help="Activation function name.",
    )

    ### self-attention GNN model args
    parser.add_argument(
        "--gin_mlp", type=str2bool, default=True, help="Whether to use MLP in GIN."
    )
    parser.add_argument(
        "--train_eps",
        type=str2bool,
        default=True,
        help="Whether to train episolon in GIN.",
    )
    parser.add_argument(
        "--edge_top_perc",
        type=float,
        default=0.2,
        help="Top fraction of edges to be kept.",
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        default="thresh",
        choices=("thresh", "knn", "thresh_abs"),
        help="Pruning method for graph.",
    )
    parser.add_argument(
        "--undirected_graph",
        type=str2bool,
        default=True,
        help="Whether make the graph undirected."
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="Absolute threshold for graph pruning.",
    )
    parser.add_argument(
        "--temporal_model",
        type=str,
        default="s4",
        choices=("gru", "s4"),
        help="Name of temporal model.",
    )
    parser.add_argument(
        "--temporal_pool",
        type=str,
        default=None,
        choices=("adaptive", "last", "mean", "pool", "first", "sum", None),
        help="Temporal pooling method",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Temporal resolution. Must be divisible by max_seq_len.",
    )
    parser.add_argument(
        "--use_prior",
        type=str2bool,
        default=False,
        help="Whether to use prior adj mat as a guide.",
    )
    parser.add_argument(
        "--negative_slope",
        type=float,
        default=0.2,
        help="Negative slope for LeakyReLU in GAT.",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=2,
        help="KNN neighbor to initialize the graph structure.",
    )
    parser.add_argument(
        "--graph_learn_metric",
        type=str,
        default="self_attention",
        choices=("self_attention", "adaptive"),
        help="Metric to learn graph structure.",
    )
    parser.add_argument(
        "--adj_embed_dim", type=int, default=16, help="Embedding dim for adaptive GSL."
    )
    parser.add_argument(
        "--regularizations",
        type=str,
        nargs="+",
        default=["feature_smoothing", "degree", "sparse"],
        choices=("feature_smoothing", "degree", "sparse"),
        help="List of regularizations to include in loss.",
    )
    parser.add_argument(
        "--residual_weight",
        type=float,
        default=0.0,
        help="Weight for residual connection in graph structure learning.",
    )
    parser.add_argument(
        "--decay_residual_weight",
        type=str2bool,
        default=False,
        help="Whether to decay the residual weight in graph structure learning."
    )
    parser.add_argument(
        "--feature_smoothing_weight",
        type=float,
        default=0.0,
        help="Loss weight for feature smoothing regularization.",
    )
    parser.add_argument(
        "--degree_weight",
        type=float,
        default=0.0,
        help="Loss weight for degree regularization.",
    )
    parser.add_argument(
        "--sparse_weight",
        type=float,
        default=0.0,
        help="Loss weight for sparsity regularization.",
    )
    # S4 model args
    parser.add_argument(
        "--bidirectional",
        type=str2bool,
        default="false",
        help="Whether or not to use bidirectional temporal model.",
    )
    parser.add_argument(
        "--state_dim", type=int, default=64, help="State dimension for S4."
    )
    parser.add_argument(
        "--prenorm",
        type=str2bool,
        default="false",
        help="Whether to add norm before S4 layer.",
    )
    parser.add_argument(
        "--postact",
        type=str,
        default=None,
        choices=(None, "glu"),
        help="Post activation in S4.",
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Channel size for S4 kernel."
    )

    ## Training/test args
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=("classification", "regression"),
        help="Model task.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=50, help="Training batch size."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of sub-processes to use per data loader.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="timm_cosine",
        choices=("cosine", "one_cycle", "timm_cosine"),
        help="LR scheduler.",
    )
    parser.add_argument(
        "--t_initial",
        type=int,
        default=100,
        help="t_initial for timm_cosine scheduler",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-5,
        help="lr_min for timm_cosine scheduler",
    )
    parser.add_argument(
        "--cycle_decay",
        type=float,
        default=0.1,
        help="cycle_decay for timm_cosine scheduler",
    )
    parser.add_argument(
        "--warmup_lr_init",
        type=float,
        default=1e-6,
        help="warmup_lr_init for timm_cosine scheduler",
    )
    parser.add_argument(
        "--warmup_t",
        type=int,
        default=5,
        help="warmup_t for timm_cosine scheduler",
    )
    parser.add_argument(
        "--cycle_limit",
        type=int,
        default=1,
        help="cycle_limit for timm_cosine scheduler",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=("adam", "adamw"),
        help="Optimizer name.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate.",
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default="auroc",
        choices=(
            "F1",
            "acc",
            "loss",
            "auroc",
            "mae",
            "rmse",
            "mse",
            "auprc",
            "kappa",
            "precision",
            "recall",
            "fbeta",
            "gbeta",
        ),
        help="Name of dev metric to determine best checkpoint.",
    )
    parser.add_argument(
        "--eval_metrics",
        type=str,
        default=[],
        nargs="+",
        choices=(
            "F1",
            "acc",
            "auroc",
            "auprc",
            "kappa",
            "rmse",
            "mae",
            "mse",
            "mape",
            "precision",
            "recall",
            "fbeta",
            "gbeta",
        ),
        help="List of metrics for evaluation of classification problems",
    )
    parser.add_argument(
        "--find_threshold_on",
        type=str,
        default=None,
        help="Which metric to maximize for cutoff thresholding."
    )
    parser.add_argument(
        "--lr_init", type=float, default="0.01", help="Initial learning rate."
    )
    parser.add_argument("--l2_wd", type=float, default=5e-3, help="L2 weight decay.")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs for which to train.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for gradient clipping.",
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="Dev/test batch size."
    )
    parser.add_argument(
        "--metric_avg",
        type=str,
        default="macro",
        help="weighted, micro, macro or binary.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Number of evaluations when eval loss is not decreasing before early stopping.",
    )
    parser.add_argument(
        "--balanced_sampling",
        default=False,
        type=str2bool,
        help="Whether to perform balanced_sampling.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help="Gradient accumulation batches.",
    )
    parser.add_argument(
        "--pos_weight",
        default=None,
        type=float,
        nargs='+',
        help="Weight for positive class in BCE loss.",
    )
    parser.add_argument(
        "--use_class_weight",
        default=False,
        type=str2bool,
        help="Whether to use class weight for cross-entropy loss.",
    )

    args = parser.parse_args()

    # which metric to maximize
    if args.metric_name in ("loss", "mae", "rmse"):
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
    elif args.metric_name in ("F1", "acc", "auroc", "auprc", "kappa"):
        # Best checkpoint is the one that maximizes F1 or acc
        args.maximize_metric = True
    else:
        raise ValueError('Unrecognized metric name: "{}"'.format(args.metric_name))

    # must provide load_model_path if testing only
    if (args.load_model_path is None) and not (args.do_train):
        raise ValueError(
            "For prediction only, please provide trained model checkpoint in argument load_model_path."
        )

    return args
