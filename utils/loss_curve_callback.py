import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os

class LossCurveCallback(pl.Callback):
    """
    A callback to record and plot training and validation loss curves.
    """
    def __init__(self, save_dir='./loss_curves'):
        """
        Args:
            save_dir (str): Directory to save the loss curve plot and data.
        """
        super().__init__()
        self.save_dir = save_dir
        self.train_losses = []
        self.val_losses = []
        
        # Create the save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called when the train epoch ends. Records the training loss.
        """
        # Get the training loss from the trainer's logger or metrics
        # We'll use the logged metrics if available
        train_loss = trainer.callback_metrics.get('train_loss') 
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        # If 'train_loss' is not directly available, you might need to log it in your model's training_step
        # For example, in your model's training_step, you should return {'loss': loss, 'train_loss': loss}
        # Or log it using self.log('train_loss', loss)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called when the validation epoch ends. Records the validation loss.
        """
        # Get the validation loss from the trainer's logger or metrics
        val_loss = trainer.callback_metrics.get('val_loss') 
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
        # Similar to training loss, ensure 'val_loss' is logged in your model's validation_step
        # For example, in your model's validation_step, you should call self.log('val_loss', loss)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called when the fit ends. Plots and saves the loss curves.
        """
        # Plotting
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        if self.train_losses:
            plt.plot(epochs, self.train_losses, label='Training Loss')
        if self.val_losses:
            # Validation might be run less frequently, so we need to align the epochs
            val_epochs = range(1, len(self.val_losses) + 1)
            # Adjust val_epochs if validation is not run every epoch
            # For simplicity, we assume validation is run every epoch
            # If not, you need to track the actual epochs when validation was run
            plt.plot(val_epochs, self.val_losses, label='Validation Loss')
            
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(self.save_dir, 'loss_curve.png')
        plt.savefig(plot_path)
        plt.close()  # Close the figure to free memory
        
        # Optionally, save the loss data to a file
        import json
        data_path = os.path.join(self.save_dir, 'loss_data.json')
        with open(data_path, 'w') as f:
            json.dump({'train_losses': self.train_losses, 'val_losses': self.val_losses}, f)
        
        print(f"Loss curve saved to {plot_path}")
        print(f"Loss data saved to {data_path}")