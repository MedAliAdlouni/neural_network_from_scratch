import matplotlib.pyplot as plt
import numpy as np

def plot_training_loss(losses):
    """
    Plot training loss over epochs with enhanced styling.

    Parameters:
        losses (list or array-like): Loss values recorded at each epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, color='royalblue', linestyle='-', marker='o', markersize=4, linewidth=1.5, label='Training Loss')

    plt.title("Training Loss Over Epochs", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

    # Add dynamic margin to y-axis limits
    min_loss, max_loss = min(losses), max(losses)
    margin = 0.05 * abs(min_loss)
    plt.ylim(min_loss - margin, max_loss + margin)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_training_loss_tuning(losses_dict):
    """
    Plot training loss curves for multiple activation functions.

    Parameters:
        losses_dict (dict): Dictionary mapping activation function names to their corresponding list of loss values.
    """
    plt.figure(figsize=(10, 6))

    for activation, losses in losses_dict.items():
        plt.plot(losses, label=activation)

    plt.title("Training Loss for Different Activation Functions", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.ylim(bottom=0)  # Ensures y-axis starts from 0
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred, activation):
    """
    Plot predicted values against true values for a given activation function.

    Parameters:
        y_true (array-like): Ground truth output values.
        y_pred (array-like): Model predictions.
        activation (str): Name of the activation function used (for plot title).
    """
    plt.figure(figsize=(8, 6))
    sample_indices = range(len(y_true))

    plt.scatter(sample_indices, y_true, color='blue', label='True Values', s=100, marker='o')
    plt.scatter(sample_indices, y_pred, color='red', label='Predicted Values', s=100, marker='x')

    plt.title(f"True vs Predicted Values ({activation})", fontsize=16, fontweight='bold')
    plt.xlabel("Sample Index", fontsize=14)
    plt.ylabel("Output Value", fontsize=14)
    plt.xticks(sample_indices)
    plt.yticks([0, 1])  # Assumes binary output; adjust as needed

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
