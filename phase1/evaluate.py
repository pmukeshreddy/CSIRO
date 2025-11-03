"""
Model evaluation and metrics.
"""
import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


class Evaluator:
    """Evaluator class for metadata predictor."""
    
    def __init__(self, model, device):
        """
        Initialize evaluator.
        
        Args:
            model: The model to evaluate
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
    
    def get_predictions(self, dataloader):
        """
        Get predictions on a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            
        Returns:
            Tuple of (predictions_dict, targets_dict)
        """
        self.model.eval()
        
        all_predictions = {
            'ndvi': [], 'height': [], 'state': [], 'species': []
        }
        all_targets = {
            'ndvi': [], 'height': [], 'state': [], 'species': []
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                images = batch['image'].to(self.device)
                
                outputs = self.model(images)
                
                # Store predictions
                all_predictions['ndvi'].extend(outputs['ndvi'].cpu().numpy())
                all_predictions['height'].extend(outputs['height'].cpu().numpy())
                all_predictions['state'].extend(
                    torch.argmax(outputs['state_logits'], dim=1).cpu().numpy()
                )
                all_predictions['species'].extend(
                    torch.argmax(outputs['species_logits'], dim=1).cpu().numpy()
                )
                
                # Store targets
                all_targets['ndvi'].extend(batch['ndvi'].numpy())
                all_targets['height'].extend(batch['height'].numpy())
                all_targets['state'].extend(batch['state'].numpy())
                all_targets['species'].extend(batch['species'].numpy())
        
        # Convert to arrays
        for key in all_predictions:
            all_predictions[key] = np.array(all_predictions[key])
            all_targets[key] = np.array(all_targets[key])
        
        return all_predictions, all_targets
    
    def calculate_metrics(self, predictions, targets):
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Regression metrics (NDVI, Height)
        for target in ['ndvi', 'height']:
            mse = mean_squared_error(targets[target], predictions[target])
            rmse = np.sqrt(mse)
            r2 = r2_score(targets[target], predictions[target])
            
            metrics[f'{target}_mse'] = mse
            metrics[f'{target}_rmse'] = rmse
            metrics[f'{target}_r2'] = r2
        
        # Classification metrics (State, Species)
        for target in ['state', 'species']:
            acc = accuracy_score(targets[target], predictions[target])
            metrics[f'{target}_accuracy'] = acc
        
        return metrics
    
    def print_metrics(self, metrics):
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        
        print("\nRegression Metrics (scaled values):")
        for target in ['ndvi', 'height']:
            print(f"\n{target.upper()}:")
            print(f"  MSE:  {metrics[f'{target}_mse']:.4f}")
            print(f"  RMSE: {metrics[f'{target}_rmse']:.4f}")
            print(f"  R²:   {metrics[f'{target}_r2']:.4f}")
        
        print("\nClassification Metrics:")
        for target in ['state', 'species']:
            acc = metrics[f'{target}_accuracy']
            print(f"\n{target.upper()}:")
            print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        
        print("\n" + "="*80)
