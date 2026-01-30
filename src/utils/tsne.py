"""
t-SNE visualization script for Swin V2 pre-trained model features.
Extracts features from multiple hierarchy levels and creates separate t-SNE plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from argparse import Namespace
from src.models.swin import SwinV2Wrapper
from src.data.dataset import ConditionsDataset
import torch.utils.data


def create_args():
    """Create args object with necessary parameters for model loading."""
    args = Namespace()
    args.img_in_channels = 7
    args.img_out_channels = 6
    args.img_size = [96, 96]
    args.swin_v2_variant = 'base'
    args.window_size = 6
    args.drop_rate = 0.0
    args.attn_drop_rate = 0.0
    args.drop_path_rate = 0.0
    args.use_light_decoder = False
    args.wandb_project = None
    args.lr = 2e-4
    args.batch_size = 16
    args.n_workers = 4
    return args


def extract_features(model, dataloader, device, num_samples=1000, region_labels=None):
    """
    Extract features from multiple hierarchy levels.
    Ensures balanced sampling across all regions.
    
    Returns:
        features_list: List of feature arrays, one per hierarchy level
        region_labels_list: List of region label arrays, one per hierarchy level
    """
    model.eval()
    model.to(device)
    
    # Disable light decoder to get features from all layers
    model.model.use_light_decoder = False
    
    # Calculate samples per region for balanced sampling
    num_regions = len(region_labels) if region_labels is not None else 1
    samples_per_region = num_samples // num_regions
    samples_collected_per_region = {i: 0 for i in range(num_regions)}
    
    print(f"Balanced sampling: {samples_per_region} samples per region (total: {num_samples})")
    
    all_features = []  # Will store list of lists (one per hierarchy)
    all_region_labels = []  # Will store region labels for each feature
    
    with torch.no_grad():
        sample_count = 0
        for batch_idx, (x, mask) in enumerate(dataloader):
            # Check if we've collected enough samples from all regions
            if all(count >= samples_per_region for count in samples_collected_per_region.values()):
                print(f"Collected samples per region: {samples_collected_per_region}")
                break
                
            x = x.to(device)
            mask = mask.to(device)
            
            # Forward pass returns (features_list, pixel_mask)
            features_list, _ = model.model(x, mask)
            
            # Initialize all_features on first batch
            if len(all_features) == 0:
                all_features = [[] for _ in range(len(features_list))]
                all_region_labels = [[] for _ in range(len(features_list))]
            
            # Get region labels for this batch
            batch_size = x.shape[0]
            if region_labels is not None:
                # Get dataset indices for this batch
                dataset_indices = dataloader.dataset.cumulative_sizes
                batch_start_idx = sample_count
                batch_region_ids = []
                for i in range(batch_size):
                    global_idx = batch_start_idx + i
                    # Find which dataset this sample belongs to
                    region_idx = 0
                    for j, cum_size in enumerate(dataset_indices):
                        if global_idx < cum_size:
                            region_idx = j
                            break
                    batch_region_ids.append(region_idx)
                batch_region_ids = np.array(batch_region_ids)
            else:
                batch_region_ids = np.zeros(batch_size, dtype=int)
            
            # Filter samples: only keep those from regions that haven't reached quota
            keep_mask = np.array([
                samples_collected_per_region[region_id] < samples_per_region
                for region_id in batch_region_ids
            ])
            
            if not np.any(keep_mask):
                # All samples in this batch are from regions that are full
                sample_count += x.shape[0]
                continue
            
            # Get filtered region IDs and update counts
            batch_region_ids_filtered = batch_region_ids[keep_mask]
            for region_id in batch_region_ids_filtered:
                samples_collected_per_region[region_id] += 1
            
            # Store features from each hierarchy level (only for samples we keep)
            for level_idx, feat in enumerate(features_list):
                # feat shape: (B, C, H, W)
                # Filter features based on keep_mask
                feat_filtered = feat[keep_mask]
                
                # Flatten spatial dimensions: (B, H*W, C)
                B, C, H, W = feat_filtered.shape
                feat_flat = feat_filtered.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
                # Convert to numpy and store
                all_features[level_idx].append(feat_flat.cpu().numpy())
                
                # Store region labels (expand to match spatial dimensions)
                region_labels_expanded = np.repeat(
                    batch_region_ids_filtered[:, None], H * W, axis=1
                )  # (B, H*W)
                all_region_labels[level_idx].append(region_labels_expanded)
            
            sample_count += x.shape[0]
    
    # Concatenate all batches for each hierarchy level
    features_concatenated = []
    region_labels_concatenated = []
    for level_idx, level_features in enumerate(all_features):
        # Stack all batches: (total_samples, H*W, C)
        level_array = np.concatenate(level_features, axis=0)
        # Reshape to (total_samples * H * W, C) for t-SNE
        total_samples, spatial, channels = level_array.shape
        level_array = level_array.reshape(total_samples * spatial, channels)
        features_concatenated.append(level_array)
        
        # Concatenate region labels
        region_array = np.concatenate(all_region_labels[level_idx], axis=0)
        region_array = region_array.reshape(total_samples * spatial)
        region_labels_concatenated.append(region_array)
    
    return features_concatenated, region_labels_concatenated


def plot_tsne(features, level_idx, output_path, n_samples=5000, region_labels=None, region_names=None):
    """
    Apply t-SNE and create visualization plot.
    
    Args:
        features: Feature array of shape (N, C)
        level_idx: Hierarchy level index (0-3)
        output_path: Path to save the plot
        n_samples: Number of samples to use for t-SNE (for speed)
        region_labels: Array of region indices for each feature (optional)
        region_names: List of region names for legend (optional)
    """
    # Subsample if too many features
    if features.shape[0] > n_samples:
        indices = np.random.choice(features.shape[0], n_samples, replace=False)
        features = features[indices]
        if region_labels is not None:
            region_labels = region_labels[indices]
    
    print(f"Computing t-SNE for hierarchy level {level_idx}...")
    print(f"  Input shape: {features.shape}")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    if region_labels is not None and region_names is not None:
        # Color-code by region
        colors = plt.cm.tab10(np.linspace(0, 1, len(region_names)))
        for region_idx, region_name in enumerate(region_names):
            mask = region_labels == region_idx
            if np.any(mask):
                plt.scatter(
                    features_2d[mask, 0], 
                    features_2d[mask, 1], 
                    alpha=0.6, 
                    s=1,
                    label=region_name,
                    c=[colors[region_idx]]
                )
        plt.legend(loc='best', fontsize=10)
    else:
        # Single color
        plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=1)
    
    plt.title(f't-SNE Visualization - Hierarchy Level {level_idx}', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {output_path}")


def main():
    # Configuration
    checkpoint_path = "saved_models/pre-trained-SwinV2/min_val_loss.ckpt"
    dataset_path = "/aspire/CarloData/CERRA-ERA5-processing/zz_processed_data/CONDITIONS"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 1000  # Number of samples to extract features from
    
    print("=" * 60)
    print("t-SNE Feature Visualization for Swin V2 Model")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print()
    
    # Create args and load model
    print("Loading model...")
    args = create_args()
    model = SwinV2Wrapper.load_from_checkpoint(checkpoint_path, args=args)
    print("Model loaded successfully!")
    print()
    
    # Create dataloader
    print("Creating dataloader...")
    
    regions = ["Iberia", "Scandinavia", "CentralEurope", "UK", "Turkey", "NordAfrica"]
    train_dataset_list = []

    for region in regions:
        dataset = ConditionsDataset(
            path=dataset_path,
            split="test",
            mask_ratio=0.0,
            model_patch_size=4,
            mask_patch_size=12,
            crop_size=96,
            region=region,
        )
        train_dataset_list.append(dataset)
    
    # Create concatenated dataset after all regions are added
    train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle to preserve region order for tracking
        num_workers=args.n_workers,
    )
        
    print(f"Dataloader created with {len(train_dataset)} total samples from {len(regions)} regions!")
    print()
    
    # Extract features
    print("Extracting features from model...")
    features_list, region_labels_list = extract_features(
        model, train_loader, device, num_samples=num_samples, region_labels=regions
    )
    print(f"Extracted features from {len(features_list)} hierarchy levels")
    print()
    
    # Create t-SNE plots for each hierarchy level
    print("Creating t-SNE visualizations...")
    for level_idx, features in enumerate(features_list):
        output_path = f"tsne_hierarchy_level_{level_idx}.png"
        plot_tsne(
            features, 
            level_idx, 
            output_path,
            region_labels=region_labels_list[level_idx],
            region_names=regions
        )
    
    print()
    print("=" * 60)
    print("All t-SNE plots created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

