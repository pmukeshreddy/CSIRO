"""
Utility functions for visualization and data exploration.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import r2_score


def visualize_sample_images(df, img_dir, num_samples=6):
    """
    Visualize sample images with metadata.
    
    Args:
        df: DataFrame containing image data
        img_dir: Directory containing images
        num_samples: Number of samples to display
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i >= len(df) or i >= num_samples:
            break
        
        row = df.iloc[i]
        img_path = os.path.join(img_dir, row['image_path'].split('/')[-1])
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            
            title = f"NDVI: {row['Pre_GSHH_NDVI']:.3f}\n"
            title += f"Height: {row['Height_Ave_cm']:.1f}cm\n"
            title += f"State: {row['State']}"
            ax.set_title(title, fontsize=9)
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Sample Pasture Images with Metadata', fontsize=14, fontweight='bold', y=1.02)
    plt.show()


def plot_categorical_distributions(df):
    """
    Plot distributions of categorical variables.
    
    Args:
        df: DataFrame containing the data
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # States
    state_counts = df['State'].value_counts()
    axes[0].bar(range(len(state_counts)), state_counts.values)
    axes[0].set_xticks(range(len(state_counts)))
    axes[0].set_xticklabels(state_counts.index, rotation=45)
    axes[0].set_title('Distribution by State', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Species (top 10)
    species_counts = df['Species'].value_counts().head(10)
    axes[1].barh(range(len(species_counts)), species_counts.values)
    axes[1].set_yticks(range(len(species_counts)))
    axes[1].set_yticklabels(
        [s[:30] + '...' if len(s) > 30 else s for s in species_counts.index],
        fontsize=9
    )
    axes[1].set_title('Top 10 Species Combinations', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Count')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nUnique states: {df['State'].nunique()}")
    print(f"Unique species combinations: {df['Species'].nunique()}")


def plot_training_curves(history, save_path='training_curves.png'):
    """
    Plot training curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(history['state_acc'], label='State Accuracy', linewidth=2)
    axes[1].plot(history['species_acc'], label='Species Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Classification Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Training curves saved to '{save_path}'")


def plot_regression_results(predictions, targets, save_path='regression_results.png'):
    """
    Plot regression predictions vs actuals.
    
    Args:
        predictions: Dictionary of predictions
        targets: Dictionary of targets
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # NDVI
    r2_ndvi = r2_score(targets['ndvi'], predictions['ndvi'])
    axes[0].scatter(targets['ndvi'], predictions['ndvi'], alpha=0.5)
    axes[0].plot(
        [targets['ndvi'].min(), targets['ndvi'].max()],
        [targets['ndvi'].min(), targets['ndvi'].max()],
        'r--', linewidth=2, label='Perfect Prediction'
    )
    axes[0].set_xlabel('Actual NDVI (scaled)', fontsize=12)
    axes[0].set_ylabel('Predicted NDVI (scaled)', fontsize=12)
    axes[0].set_title(f'NDVI Predictions (R²={r2_ndvi:.4f})', 
                      fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Height
    r2_height = r2_score(targets['height'], predictions['height'])
    axes[1].scatter(targets['height'], predictions['height'], alpha=0.5)
    axes[1].plot(
        [targets['height'].min(), targets['height'].max()],
        [targets['height'].min(), targets['height'].max()],
        'r--', linewidth=2, label='Perfect Prediction'
    )
    axes[1].set_xlabel('Actual Height (scaled)', fontsize=12)
    axes[1].set_ylabel('Predicted Height (scaled)', fontsize=12)
    axes[1].set_title(f'Height Predictions (R²={r2_height:.4f})',
                      fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Regression results saved to '{save_path}'")


def print_summary(best_val_loss, metrics, config):
    """
    Print training summary.
    
    Args:
        best_val_loss: Best validation loss achieved
        metrics: Dictionary of evaluation metrics
        config: Configuration object
    """
    print("\n" + "="*80)
    print("STAGE 1 COMPLETE! 🎉")
    print("="*80)
    
    print("\n📊 Final Results:")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"  NDVI R²: {metrics['ndvi_r2']:.4f}")
    print(f"  Height R²: {metrics['height_r2']:.4f}")
    print(f"  State Accuracy: {metrics['state_accuracy']:.4f}")
    print(f"  Species Accuracy: {metrics['species_accuracy']:.4f}")
    
    print(f"\n💾 Saved Files:")
    print(f"  Model: {config.MODEL_PATH}")
    print(f"  Preprocessor: {config.PREPROCESSOR_PATH}")
    print(f"  Training curves: training_curves.png")
    print(f"  Results: regression_results.png")
    
    print("\n🚀 Next Steps:")
    print("  1. Review the results above")
    print("  2. If performance is good (R² > 0.7, Acc > 0.6), proceed to Stage 2")
    print("  3. If not, try:")
    print("     - Increase epochs")
    print("     - Adjust learning rate")
    print("     - Add more data augmentation")
    print("     - Try different backbone (EfficientNet, etc.)")
    
    print("\n" + "="*80)
