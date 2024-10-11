import pandas as pd
import matplotlib.pyplot as plt

# Load the trials training metrics and trials params CSV files
metrics_file = 'trials_training_metrics.csv'
params_file = 'trials_params.csv'

# Read the CSV files into DataFrames
metrics_df = pd.read_csv(metrics_file).reset_index(drop=True)
params_df = pd.read_csv(params_file).reset_index(drop=True)

# Find the best eval metrics per trial (assuming best = highest eval_reassembly)
best_metrics_df = metrics_df.groupby(['trial_number', 'lr_scheduler']).apply(lambda group: group.loc[group['eval_reassembly'].idxmax()]).reset_index(drop=True)

# Select relevant eval metrics to merge into the params_df
# Columns can be adjusted to what you need from the metrics file
metrics_cols = [c for c in best_metrics_df.columns if c.startswith('eval_')]
best_eval_metrics = best_metrics_df[['trial_number', 'lr_scheduler'] + metrics_cols]

# Merge the best eval metrics into the params_df based on trial_number
updated_params_df = pd.merge(params_df, best_eval_metrics, on=['trial_number', 'lr_scheduler'], how='left')

# Save the updated trials_params.csv
updated_params_df.to_csv('trials_params_updated.csv', index=False)

print("Updated 'trials_params.csv' with best eval metrics saved as 'trials_params_updated.csv'.")

# Function to generate the requested plots for each scheduler
def plot_training_evaluation_curves(scheduler_name):
    # Filter the dataframe for the specific scheduler
    scheduler_df = updated_params_df[updated_params_df['lr_scheduler'] == scheduler_name]

    # Get the trial with the highest eval_reassembly
    best_trial_number = scheduler_df.loc[scheduler_df['eval_reassembly'].idxmax()]['trial_number']
    print(f'Best trial number for {scheduler_name}: {best_trial_number}')
    best_trial_df = metrics_df[metrics_df['trial_number'] == best_trial_number].sort_values(by='epoch')

    # print(best_trial_df[['epoch', 'eval_reassembly', 'loss', 'eval_loss', 'learning_rate']].to_markdown())

    print(f'Best trial for {scheduler_name}: {best_trial_number}')
    # Get the row with the highest eval_reassembly
    best_trial_row = scheduler_df.loc[scheduler_df['trial_number'] == best_trial_number]
    for col in best_trial_row.columns:
        if pd.isnull(best_trial_row[col].values[0]):
            continue
        if isinstance(best_trial_row[col].values[0], int) or isinstance(best_trial_row[col].values[0], str):
            print(f'\t- {col}: {best_trial_row[col].values[0]}')
        elif 'learning_rate' in col or '_lr' in col:
            print(f'\t- {col}: {best_trial_row[col].values[0]:.2e}')
        else:
            print(f'\t- {col}: {best_trial_row[col].values[0]:.4f}')
    print('-' * 80)

    # Create a 3-subplot figure for the best trial
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Subplot 1: Eval reassembly
    tmp = best_trial_df.dropna(subset=['eval_reassembly'])
    # print(tmp[['epoch', 'eval_reassembly']].sort_values(by='epoch').to_markdown())
    # print('-' * 80)
    axs[0].plot(tmp['epoch'], tmp['eval_reassembly'], color='C0', label='Eval Reassembly')
    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel('Eval Reassembly')
    axs[0].legend(loc='upper left')
    axs[0].set_title(f'Training and Evaluation Curves - Best Trial (Scheduler: {scheduler_name})')
    axs[0].grid(axis='both', alpha=0.5)

    # Subplot 2: Learning rate
    tmp = best_trial_df.dropna(subset=['learning_rate'])
    # print(tmp[['epoch', 'learning_rate']].sort_values(by='epoch').to_markdown())
    # print('-' * 80)
    axs[1].plot(tmp['epoch'], tmp['learning_rate'], color='C1', label='Learning Rate')
    axs[1].set_ylabel('Learning Rate')
    axs[1].legend(loc='upper left')
    axs[1].grid(axis='both', alpha=0.5)

    # Subplot 3: Train and eval loss
    tmp = best_trial_df.dropna(subset=['loss'])
    axs[2].plot(tmp['epoch'], tmp['loss'], color='C2', label='Train Loss')
    tmp = best_trial_df.dropna(subset=['eval_loss'])
    axs[2].plot(tmp['epoch'], tmp['eval_loss'], color='C3', label='Eval Loss')
    axs[2].set_ylabel('Loss')
    axs[2].set_xlabel('Epoch')
    axs[2].legend(loc='upper left')
    axs[2].grid(axis='both', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'logs/training_evaluation_curves_{scheduler_name}.pdf', bbox_inches='tight')
    plt.close()


lr_schedulers = ['cosine', 'cosine_restarts', 'reduce_lr']

# Generate plots for each scheduler
for scheduler in lr_schedulers:
    # print(f"Generating training and evaluation curves for scheduler: {scheduler}")
    plot_training_evaluation_curves(scheduler)
