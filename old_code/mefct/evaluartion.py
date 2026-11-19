import numpy as np
import glob
import matplotlib.pyplot as plt


# Specify the directory where histories are saved and the number of folds
history_dir = '/cluster/home/mrostamian/log_scrach/'
num_folds = 5  # Update based on the number of folds you have

# Initialize dictionary to store metrics for each fold
all_metrics = {}
all_metrics_plot = {}

# Loop through each fold and load the respective history file
for fold in range(num_folds):
    fold_file = f"{history_dir}brats_history_fold_{fold}.npy"
    history = np.load(fold_file, allow_pickle=True).item()  # Load history for this fold
    
    # Store metrics for each fold
    for metric, values in history.items():
        if metric not in all_metrics:
            all_metrics[metric] = []
            all_metrics_plot[metric] = []
        all_metrics[metric].append(values[-1])  # Get the final metric value for this fold
        all_metrics_plot[metric].append(values)  # Store all values for plotting


# Compute averages across folds
average_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}
std_dev_metrics = {metric: np.std(values) for metric, values in all_metrics.items()}

# Display results
print("Results per Fold:")
for metric, values in all_metrics.items():
    print(f"{metric}: {values}")

print("\nAverage Results across Folds:")
for metric, avg in average_metrics.items():
    print(f"{metric}: {avg:.4f} ± {std_dev_metrics[metric]:.4f}")


# # Check for missing metrics and handle gracefully
# for metric in list(all_metrics_plot.keys()):
#     if len(all_metrics_plot[metric]) < num_folds:
#         print(f"Warning: Metric '{metric}' does not have values for all folds. Skipping this metric.")
#         del all_metrics[metric]  # Remove metrics that do not have all folds' data



# Plot metrics for each metric separately
for metric, values in all_metrics_plot.items():
    plt.figure(figsize=(10, 6))
    for fold in range(num_folds):
        plt.plot(values[fold], label=f'Fold {fold}', alpha=0.7)  # Plot each fold's values
    plt.title(f'{metric} across Folds')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    plt.savefig(f"{history_dir}{metric}_folds_plot.png")  # Save the plot as a PNG file
    plt.show()  # Display the plot