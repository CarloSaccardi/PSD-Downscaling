# Standard library
import random
import time
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed

# First-party
from src import constants, utils
from src.models import UNetWrapper, DiffusionWrapper
from src.data import ERA5CERRAtiled
import os
import yaml

# NEW: Import new configuration system (optional)
try:
    import sys
    from pathlib import Path
    src_path = str(Path(__file__).parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from config.base_config import ExperimentConfig  # type: ignore
    from config.legacy_compat import convert_legacy_config_to_new, create_legacy_args_from_config  # type: ignore
    NEW_CONFIG_AVAILABLE = True
except ImportError:
    NEW_CONFIG_AVAILABLE = False

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODELS = {
    "UNet-CNN": UNetWrapper,
    "Diffusion": DiffusionWrapper
}


def get_args():
    """
    Main function for training and evaluating models
    """
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )
    # General options
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., yaml_configs/good_runs/UNet/UNet_test.yaml)",
    )
    parser.add_argument(
        "--dataset_cerra",
        type=str,
        default="/aspire/CarloData/MASK_GNN_DATA/CERRA_interpolated_300x300",
        help="Dataset, corresponding to name in data directory "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--dataset_era5",
        type=str,
        default="/aspire/CarloData/MASK_GNN_DATA/ERA5_60_n2_40_18",
        help="Dataset, corresponding to name in data directory "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_efm",
        help="Model architecture to train/evaluate (default: graph_lam)",
    )
    parser.add_argument(
        "--subset_ds",
        type=int,
        default=0,
        help="Use only a small subset of the dataset, for debugging"
        "(default: 0=false)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        type=int,
        default=0,
        help="If optimizer state should be restored with model "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )
    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="neural-lam",
        help="Wandb project name",
    )
    parser.add_argument(
        "--run_name",  
        type=str,
        default=None,
        help="Name of the run",
    )
    ########################################################
    # DATASET #
    parser.add_argument(
        "--output_variables",  
        type=list,
        default=None,
        help="List of output variables to predict",
    )
    parser.add_argument(
        "--img_in_channels",  
        type=int,
        default=5,
        help="Number of input channels",
    )
    parser.add_argument(
        "--img_out_channels",  
        type=int,
        default=5,
        help="Number of output channels",
    )
    parser.add_argument(
        "--img_resolution",  
        type=list,
        default=[300, 300],
        help="Resolution of the input images",
    )
    ########################################################
    # MODEL #
    parser.add_argument(
        "--N_grid_channels",  
        type=int,
        default=4,
        help="Number of grid channels",
    )
    parser.add_argument(
        "--embedding_type",  
        type=str,
        default="zero",
        help="List of output variables to predict",
    )
    parser.add_argument(
        "--model_channels",  
        type=int,
        default=64,
        help="Number of model channels",
    )
    parser.add_argument(
        "--channel_mult",  
        type=list,
        default=[1, 2, 2],
        help="List of channel multipliers",
    )
    parser.add_argument(
        "--attn_resolutions",  
        type=list,
        default=[16],
        help="List of attention resolutions",
    )
    parser.add_argument(
        "--model_type",  
        type=str,
        default="SongUNetPosEmbd",
        help="List of attention resolutions",
    )
    ########################################################
    # TRAINING #
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--anneal_epochs",
        type=int,
        default=200,
        help="number of epochs to anneal lambda (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--lr_decay", type=int, default=1, help="learning rate decay (default: 1)"
    )
    parser.add_argument(
        "--lr_rampup", type=int, default=0, help="learning rate rampup (default: 0)"
    )
    parser.add_argument(
        "--grad_clip_threshold", type=int, default=None, help="gradient clipping threshold (default: None)"
    )
    parser.add_argument(
        "--checkpoint_level",
        type=int,
        default=0,
        help="Checkpoint level for the model (default: 1)",
    )
    parser.add_argument(
        "--regression_net",
        type=str,
        help="Path to load model parameters from regression step.",
    )
    parser.add_argument(
        "--gridtype",
        type=str,
        help="Type of positional grid to use: 'sinusoidal', 'learnable', 'linear', or 'test'.",
    )
    #args.hr_mean_conditioning
    parser.add_argument(
        "--hr_mean_conditioning",
        type=bool,
        default=True,
        help="Condition on regression model prediction or not",
    )
    parser.add_argument(
        "--num_ensembles",
        type=int,
        default=32,
        help="Number of ensembles to generate with diffusion",
    )    
    parser.add_argument(
        "--savepreds_path",
        type=str,
        default=None,
        help="Path to save predictions to",
    )
    parser.add_argument(
        "--lambda_psd",
        type=float,
        default=0.1,
        help="Weight for the PSD loss term (default: 0.0)",
    )
    parser.add_argument(
        "--init_lambda",
        type=float,
        default=0.0,
        help="Weight for the PSD loss term (default: 0.0)",
    )
    parser.add_argument(
        "--max_lambda",
        type=float,
        default=0.1,
        help="Weight for the PSD loss term (default: 0.0)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        help="Type of loss to use (default: None --> MSE)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to resume training from (default: None)",
    )
    
    # NEW: Add new configuration system option
    if NEW_CONFIG_AVAILABLE:
        parser.add_argument(
            "--use_new_config",
            action="store_true",
            help="Use new configuration system with validation (if available)"
        )
        parser.add_argument(
            "--validate_config",
            action="store_true", 
            help="Validate configuration and exit (useful for testing configs)"
        )
    
    return parser.parse_args()

def main(args):
    # Convert legacy args to new config system
    print("🔧 Using new configuration system with validation...")
    
    try:
        config = convert_legacy_config_to_new(args)
        print("✅ Configuration validation passed!")
        
        # If only validating, exit here
        if hasattr(args, 'validate_config') and args.validate_config:
            print("✅ Configuration is valid. Exiting as requested.")
            return
        
        print("🔄 Using new configuration system directly")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        print("💡 Tip: Check your YAML file for errors or use --help for more options")
        return
    
    # Asserts for configuration
    assert config.model.model_type in MODELS, f"Unknown model: {config.model.model_type}"
    assert config.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {config.eval}"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)
    devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    print(f"Using {devices} GPUs")

    # Set seed
    seed.seed_everything(config.training.seed)

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        #torch.set_float32_matmul_precision(
        #    "high"
        #)  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    # Load model parameters
    model_class = MODELS[config.model.model_type]
    if config.load:
        # For now, we still need to pass args to the model for backward compatibility
        # This will be updated when we migrate the model classes
        legacy_args = create_legacy_args_from_config(config)
        model = model_class.load_from_checkpoint(config.load, args=legacy_args)
        if config.restore_opt:
            # Save for later
            # Unclear if this works for multi-GPU
            model.opt_state = torch.load(config.load)["optimizer_states"][0]
    else:
        legacy_args = create_legacy_args_from_config(config)
        model = model_class(legacy_args)

    prefix = "subset-" if config.dataset.subset_size else ""
    prefix += config.run_name if config.run_name else ""
    if config.eval:
        prefix = prefix + f"eval-{config.eval}-"
    run_name = (
        f"{prefix}-{config.model.model_type}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )

    # Callbacks for saving model checkpoint
    callbacks = []
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            filename="min_val_loss",
            monitor="val_loss",
            mode="min",
            save_last=True,
        )
    )
    
    if config.wandb_project is not None:
        logger = pl.loggers.WandbLogger(
            project=config.wandb_project, name=run_name, config=config
        )
    else:
        logger = pl.loggers.TensorBoardLogger(
            save_dir="DebugLogs/", name=run_name
        )  # or CSVLogger


    # Training strategy
    # If doing pure autoencoder training (kl_beta = 0), the prior network is not
    # used at all in producing the loss. This is desired, but DDP complains.
    strategy = "ddp"

    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        deterministic=True,
        strategy=strategy,
        accelerator=device_name,
        #devices=devices,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=config.training.val_interval,
        precision=config.training.precision,
        #profiler="simple",
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0 and isinstance(logger, pl.loggers.WandbLogger):
        utils.init_wandb_metrics(logger)  # Do after wandb.init

    if config.eval:
        eval_loader = torch.utils.data.DataLoader(
            ERA5CERRAtiled(
                config.dataset.cerra_path,
                config.dataset.era5_path,
                split="test",#TODO: Change to val
                subset=False,
            ),
            config.training.batch_size,
            shuffle=False,
            num_workers=config.training.n_workers,
        )

        print(f"Running evaluation on {config.eval}")
        trainer.test(model=model, dataloaders=eval_loader)
    else:
        # Load data
        train_loader = torch.utils.data.DataLoader(
            ERA5CERRAtiled(
                config.dataset.cerra_path,
                config.dataset.era5_path,
                split="train",
                subset=bool(config.dataset.subset_size),
            ),
            config.training.batch_size,
            shuffle=True,
            num_workers=config.training.n_workers,
        )
        
        val_loader = torch.utils.data.DataLoader(
            ERA5CERRAtiled(
                config.dataset.cerra_path,
                config.dataset.era5_path,
                split="val",
                subset=bool(config.dataset.subset_size),
            ),
            config.training.batch_size,
            shuffle=False,
            num_workers=config.training.n_workers,
        )
        # Train model
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=config.resume if config.resume else None,
        )


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)  
  

if __name__ == '__main__':
    
    args = get_args()

    if args.config is not None:

        with open(str(args.config), "r") as f:
            config = yaml.safe_load(f)
        update_args(args, config)

    main(args)
