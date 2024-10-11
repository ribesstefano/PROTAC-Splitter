import os
import json
import csv

# Define the root directory and the three projects
root_dir = '/mimer/NOBACKUP/groups/naiss2023-6-290/stefano/models/'
projects = ['cosine', 'cosine_restarts', 'reduce_lr']

# Define the output CSV file
output_file = 'trials_params.csv'

# Initialize an empty list to store trial data
trial_data = []

# Function to extract the highest checkpoint directory
def get_highest_checkpoint(trial_dir):
    checkpoint_dirs = [d for d in os.listdir(trial_dir) if d.startswith('checkpoint-')]
    # Sort the checkpoint directories by number and return the highest
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]), reverse=True)
    return checkpoint_dirs[0] if checkpoint_dirs else None

# Function to extract trial parameters from the trainer_state.json
def extract_trial_params(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    trial_params = data.get('trial_params', {})
    return trial_params

# A set to track all unique keys (parameters) encountered
all_keys = set()

# Loop over each project
for project in projects:
    project_dir = os.path.join(root_dir, f'PROTAC-Splitter-standard_recombined-ChemBERTa-lr_{project}-opt25')
    
    # Loop over all trial directories
    for trial_dir in os.listdir(project_dir):
        if trial_dir.startswith('trial-number='):
            trial_path = os.path.join(project_dir, trial_dir)
            
            # Get the highest checkpoint directory
            highest_checkpoint = get_highest_checkpoint(trial_path)
            if highest_checkpoint:
                checkpoint_path = os.path.join(trial_path, highest_checkpoint)
                
                # Locate the trainer_state.json file
                trainer_state_path = os.path.join(checkpoint_path, 'trainer_state.json')
                if os.path.exists(trainer_state_path):
                    
                    # Extract trial parameters
                    trial_params = extract_trial_params(trainer_state_path)
                    
                    # Extract trial number
                    trial_number = trial_dir.split('trial-number=')[1].split('-')[0]
                    
                    # Add project and trial number to the parameters
                    trial_entry = {
                        'lr_scheduler': project,
                        'trial_number': trial_number
                    }
                    trial_entry.update(trial_params)
                    
                    # Add trial data to the list
                    trial_data.append(trial_entry)
                    
                    # Track all keys (parameters)
                    all_keys.update(trial_entry.keys())

# Write the trial data to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    # Define the fieldnames dynamically based on all encountered keys
    # fieldnames = sorted(all_keys)
    all_keys.remove('trial_number')
    all_keys.remove('lr_scheduler')
    fieldnames = ['trial_number', 'lr_scheduler'] + list(all_keys)
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()
    
    # Write the rows
    for trial in trial_data:
        writer.writerow(trial)

print(f"Data collected and saved to {output_file}")
