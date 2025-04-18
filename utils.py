import matplotlib.pyplot as plt
import numpy as np

# Function to plot training loss with enhanced styling
def plot_training_loss(losses):
    """
    Plots the training loss over epochs with enhanced styling.

    Parameters:
        losses (list or array-like): A list or array of loss values per epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, color='royalblue', linestyle='-', marker='o', markersize=4, linewidth=1.5, label='Training Loss')
    
    # Enhancing the plot
    plt.title("Training Loss Over Epochs", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    
    # Dynamic y-axis limits with a margin
    min_loss = min(losses)
    max_loss = max(losses)
    plt.ylim(min_loss - 0.05 * abs(min_loss), max_loss + 0.05 * abs(max_loss))
    
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    
    # Adding tight layout for better spacing
    plt.tight_layout()
    
    # Display plot
    plt.show()


# Function to plot training loss for all activation functions
def plot_training_loss(losses_dict):
    plt.figure(figsize=(10, 6))
    
    for activation, losses in losses_dict.items():
        plt.plot(losses, label=activation)  # Plot each activation's loss

    plt.title('Training Loss for Different Activation Functions')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.ylim(bottom=0)  # Optional: adjust y-axis to start from 0
    plt.show()

# Function to plot predictions against true values
def plot_predictions(y_true, y_pred, activation):
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_true)), y_true, color='blue', label='True Values', s=100, marker='o')
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Values', s=100, marker='x')
    plt.title(f'True vs Predicted Values ({activation})')
    plt.xlabel('Sample Index')
    plt.ylabel('Output Value')
    plt.xticks(range(len(y_true)))
    plt.yticks([0, 1])  # Adjust as needed for your output range
    plt.legend()
    plt.grid()
    plt.show()
