import argparse
import json
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from data_loader.function_dataset import FunctionDataLoader
from model.dynamic_model import DynamicModel
from model.metric import mse_loss, mae_loss, r2_score, explained_variance

def main(config_name):
    # Load configuration
    config_path = Path('configs') / f'{config_name}.json'
    with open(config_path) as f:
        config = json.load(f)
    
    # Create data loader
    data_args = config['data_loader']['args']
    data_loader = FunctionDataLoader(**data_args)
    
    # Create model
    model = DynamicModel(**config['arch']['args'])
    
    # Setup optimizer
    optimizer_class = getattr(torch.optim, config['optimizer']['type'])
    optimizer = optimizer_class(model.parameters(), **config['optimizer']['args'])
    
    # Setup learning rate scheduler
    scheduler_class = getattr(torch.optim.lr_scheduler, config['lr_scheduler']['type'])
    scheduler = scheduler_class(optimizer, **config['lr_scheduler']['args'])
    
    # Setup tensorboard
    writer = SummaryWriter(f'runs/{config_name}')
    
    # Training loop
    epochs = config['trainer']['epochs']
    best_val_loss = float('inf')
    early_stop_count = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.MSELoss()(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_metrics = {'r2': 0.0, 'mae': 0.0, 'exp_var': 0.0}
        num_val_batches = 0
        
        with torch.no_grad():
            for x, y in data_loader.split_validation:
                output = model(x)
                val_loss += mse_loss(output, y)
                val_metrics['r2'] += r2_score(output, y)
                val_metrics['mae'] += mae_loss(output, y)
                val_metrics['exp_var'] += explained_variance(output, y)
                num_val_batches += 1
        
        # Calculate average metrics
        train_loss = epoch_loss / len(data_loader)
        val_loss = val_loss / num_val_batches
        for k in val_metrics:
            val_metrics[k] /= num_val_batches
        
        # Update learning rate scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Metrics/R2', val_metrics['r2'], epoch)
        writer.add_scalar('Metrics/MAE', val_metrics['mae'], epoch)
        writer.add_scalar('Metrics/ExpVar', val_metrics['exp_var'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            # Save best model
            torch.save(model.state_dict(), f'saved/{config_name}_best.pth')
        else:
            early_stop_count += 1
            if early_stop_count >= config['trainer']['early_stop']:
                print(f'Early stopping at epoch {epoch}')
                break
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, '
                  f'r2={val_metrics["r2"]:.4f}')
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config', help='config name (without .json)')
    args = parser.parse_args()
    main(args.config)