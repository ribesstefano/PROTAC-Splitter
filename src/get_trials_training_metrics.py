import re
import csv
import os

# Define the root directory containing the log files
log_dir = './'
log_pattern = 'slurm-train-PROTAC-Splitter-standard_recombined-ChemBERTa-lr_.*\.log'

# Define the output CSV file
output_file = 'trials_training_metrics.csv'

# Regular expression patterns to capture relevant metrics
loss_pattern = re.compile(r"'loss': ([\d\.]+)")
grad_norm_pattern = re.compile(r"'grad_norm': ([\d\.]+)")
learning_rate_pattern = re.compile(r"'learning_rate': ([\de\-\.]+)")
eval_loss_pattern = re.compile(r"'eval_loss': ([\d\.]+)")
eval_valid_pattern = re.compile(r"'eval_valid': ([\d\.]+)")
epoch_pattern = re.compile(r"'epoch': ([\d\.]+)")
eval_runtime_pattern = re.compile(r"'eval_runtime': ([\d\.]+)")
eval_samples_pattern = re.compile(r"'eval_samples_per_second': ([\d\.]+)")
eval_steps_pattern = re.compile(r"'eval_steps_per_second': ([\d\.]+)")
parameters_pattern = re.compile(r"Number of parameters: 102,393,599")

# {'eval_loss': 0.2445475310087204, 'eval_has_three_substructures': 1.0, 'eval_has_all_attachment_points': 1.0, 'eval_valid': 1.0, 'eval_reassembly': 0.0, 'eval_e3_valid': 1.0, 'eval_e3_has_attachment_point(s)': 1.0, 'eval_poi_valid': 1.0, 'eval_poi_has_attachment_point(s)': 1.0, 'eval_linker_valid': 1.0, 'eval_linker_has_attachment_point(s)': 1.0, 'eval_rouge1': 0.5100280336935934, 'eval_rouge2': 0.14818530356266457, 'eval_rougeL': 0.38502415507981946, 'eval_rougeLsum': 0.3851565459043593, 'eval_runtime': 26.5219, 'eval_samples_per_second': 11.802, 'eval_steps_per_second': 0.189, 'epoch': 0.128}



# Parameters for step increments
logging_steps = 500
eval_steps = 1000

# Initialize an empty list to store the extracted metrics
metrics_data = []

# Function to extract metrics from a line
def extract_metrics(line):
    metrics = {}
    
    # Search for different metrics using regular expressions
    loss_match = loss_pattern.search(line)
    grad_norm_match = grad_norm_pattern.search(line)
    learning_rate_match = learning_rate_pattern.search(line)
    eval_loss_match = eval_loss_pattern.search(line)
    eval_valid_match = eval_valid_pattern.search(line)
    epoch_match = epoch_pattern.search(line)
    eval_runtime_match = eval_runtime_pattern.search(line)
    eval_samples_match = eval_samples_pattern.search(line)
    eval_steps_match = eval_steps_pattern.search(line)

    if loss_match:
        metrics['loss'] = float(loss_match.group(1))
    if grad_norm_match:
        metrics['grad_norm'] = float(grad_norm_match.group(1))
    if learning_rate_match:
        metrics['learning_rate'] = float(learning_rate_match.group(1))
    if eval_loss_match:
        metrics['eval_loss'] = float(eval_loss_match.group(1))
    if epoch_match:
        metrics['epoch'] = float(epoch_match.group(1))
    if eval_runtime_match:
        metrics['eval_runtime'] = float(eval_runtime_match.group(1))
    if eval_samples_match:
        metrics['eval_samples_per_second'] = float(eval_samples_match.group(1))
    if eval_steps_match:
        metrics['eval_steps_per_second'] = float(eval_steps_match.group(1))

    if (learning_rate_match and epoch_match) or (eval_loss_match and eval_valid_match):
        metrics = eval(line)

    return metrics if metrics else None

# Loop through all log files in the directory
for lr_scheduler in ['cosine', 'cosine_restarts', 'reduce_lr']:
    log_file = f'slurm-train-PROTAC-Splitter-standard_recombined-ChemBERTa-lr_{lr_scheduler}-opt25.log'
    log_path = os.path.join(log_dir, log_file)
    with open(log_path, 'r') as f:
        trial_number = -1
        train_step = 0
        eval_step = 0

        # NOTE: Skip the first 2 lines with `parameters_pattern` in them
        skipped_lines = 2

        for line in f:
            # Detect start of a new trial after parameter count
            if parameters_pattern.search(line):
                if skipped_lines > 0:
                    skipped_lines -= 1
                    continue
                trial_number += 1
                train_step = 0  # Reset train and eval steps for each trial
                eval_step = 0
                continue
            
            if trial_number < 0:
                continue

            # Extract and store metrics
            metrics = extract_metrics(line)
            if metrics:
                metrics['trial_number'] = trial_number # Start from 0
                metrics['train_step'] = train_step
                metrics['eval_step'] = eval_step
                metrics['lr_scheduler'] = lr_scheduler
                
                # Increment steps
                if 'loss' in metrics:
                    train_step += logging_steps
                if 'eval_loss' in metrics:
                    eval_step += eval_steps
                
                metrics_data.append(metrics)

# Extract the fieldnames dynamically from the metrics keys
all_fieldnames = set()
for metrics in metrics_data:
    all_fieldnames.update(metrics.keys())

# Write the collected metrics to the CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=sorted(all_fieldnames))
    
    # Write the header
    writer.writeheader()
    
    # Write the rows
    for metrics in metrics_data:
        writer.writerow(metrics)

print(f"Metrics collected and saved to {output_file}")
