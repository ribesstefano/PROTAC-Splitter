import os
import argparse
from typing import Optional

import seaborn as sns
import pandas as pd
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
# from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.dummy import DummyClassifier
from rdkit import Chem
from rdkit.Chem import Draw

from protac_splitter.display_utils import (
    safe_display,
    get_mapped_protac_img,
    display_mol,
)
from protac_splitter.evaluation import split_prediction


def process_score_file(csv_file, top_metric='reassembly'):
    """
    Processes a list of CSV files to extract model scores, normalize them, 
    and organize them for heatmap plotting per generation strategy.

    Parameters:
    - csv_file: file paths to CSV score files.
    - max_metrics: list of metrics where higher is better.
    - min_metrics: list of metrics where lower is better.
    - zero_metrics: list of metrics where closer to zero is better.

    Returns:
    - dict of DataFrames: {generation_strategy: normalized DataFrame of scores}
    """

    # Dictionary to store results per generation strategy
    final_df = []

    df = pd.read_csv(csv_file)
    print('-' * 80)
    print(f"Processing {csv_file}")
    print('-' * 80)
    
    # Extract model name
    model_name = df["model_name"].iloc[0] if "model_name" in df.columns else csv_file
    
    # Identify all generation strategies based on column names
    strategies = set(col.split("_pred_n")[0] for col in df.columns if "_pred_n" in col)
    
    print(f"Processing {model_name} with {len(strategies)} strategies")
    
    for strategy in strategies:
        # Extract all relevant metric columns for this strategy
        strategy_columns = [col for col in df.columns if col.startswith(strategy + "_pred_n")]

        # Construct a set of all metrics that have "_pred_nN_" in their name
        num_preds = len(set(col.split("_pred_n")[1][0] for col in strategy_columns))

        metric_names = set(col.split("_pred_n")[1][2:] for col in strategy_columns)

        # Remove the "empty" metric name, associated to the prediction SMILES
        metric_names.discard("")
        
        if "search" in strategy or num_preds > 1:
            # Among the predictions for a given strategy, select the best one based on the top_metric argument
            best_pred = df[[f"{strategy}_pred_n{i}_{top_metric}" for i in range(num_preds)]].idxmax(axis=1)
            
            # Extract the index of the best prediction
            best_pred_idx = best_pred.apply(lambda x: int(x.split("_pred_n")[-1][0]))

            # Add the indexes to the DataFrame
            df['best_pred_idx'] = best_pred_idx

        for metric in metric_names:
            top_1 = df[strategy + "_pred_n0_" + metric]

            if num_preds > 1:
                top_5 = df.apply(lambda x: x[strategy + f"_pred_n{x['best_pred_idx']}_" + metric], axis=1)
            else:
                top_5 = None

            row = {
                'model_name': model_name,
                'strategy': strategy,
                'metric': metric,
                'top_n': 1,
                'fixed': True if 'fixed' in metric else False,
            }
            for reduce in ['mean', 'max', 'min', 'std']:
                row['reduce'] = reduce
                row['value'] = top_1.dropna().agg(reduce)
                final_df.append(row.copy())
            
            if top_5 is not None:
                row = {
                    'model_name': model_name,
                    'strategy': strategy,
                    'metric': metric,
                    'top_n': num_preds,
                    'fixed': True if 'fixed' in metric else False,
                }
                for reduce in ['mean', 'max', 'min', 'std']:
                    row['reduce'] = reduce
                    row['value'] = top_5.dropna().agg(reduce)
                    final_df.append(row.copy())

    return pd.DataFrame(final_df)

def main(
    score_file: Optional[str] = "logs/PROTAC-Splitter-Model-v2-scores.csv",
    best_strategy: str = 'beam_search_decoding',
    img_dir: Optional[str] = None,
    failed_metric_to_display: str = "default_pred_n0_reassembly_fixed",
    print_failed_predictions: bool = False,
):
    # Check if the image directory exists, if not create it
    if img_dir is not None:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        
    scores_df = pd.read_csv(score_file)
    processed_scores_df = process_score_file(score_file)

    if processed_scores_df['strategy'].nunique() > 1:
        print('-' * 80)
        print('Compare generation strategies')
        print('-' * 80)
        for metric in ['valid', 'reassembly', 'reassembly_nostereo', 'all_ligands_equal', 'heavy_atoms_difference']:
            df = processed_scores_df.copy()
            
            # Isolate the generation strategy
            df = df[(df['metric'] == metric) | (df['metric'] == metric + "_fixed")]
            df = df[df['reduce'] == 'mean']
            df = df[df['top_n'] == 1]
            df['value'] = df['value'].astype(float)

            # Title the strategies
            df['strategy'] = df['strategy'].str.split("_").str.join(" ").str.title()
            # Rename the "fixed" column from True/False to "Fixed"/"Not Fixed"
            df['fixed'] = df['fixed'].replace({True: "Fixed", False: "Not Fixed"})
            
            # Bar plot the "value" on the x-axis, "strategy" on the y-axis and "fixed" as hue
            plt.figure(figsize=(8, 6))
            ax = sns.barplot(data=df, y='strategy', x='value', hue='fixed')
            metric_title = " ".join(metric.split("_")).title()
            plt.title(f"{metric_title} - Top-1")
            plt.ylabel("")
            plt.xlabel("")
            plt.grid(axis='x', alpha=0.5)
            
            # Draw the values on the bars at the center
            for p in ax.patches:
                if p.get_width() == 0:
                    continue
                if metric != 'heavy_atoms_difference':
                    text = f"{p.get_width():.2%}"
                else:
                    text = f"{p.get_width():.2f}"
                ax.annotate(text, (0.05, p.get_y() + p.get_height() / 2),
                                ha='left', va='center', fontsize=10, color='white')
            
            # If metric is not "heavy_atoms_difference", set the x-axis to be a percentage
            if metric != 'heavy_atoms_difference':
                plt.xlim(0, 1.05)
                plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))
            # Place the legend outside the plot, on the bottom with two columns. Then remove the title
            plt.legend(title=None, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
            
            if img_dir:
                plot_file = os.path.join(img_dir, f"plot_{metric}_top1.pdf")
                plt.savefig(plot_file, bbox_inches='tight')
                plt.clf()
                plt.close()
                print(f"Plot saved to: {plot_file}")
            else:
                plt.show()


    print('-' * 80)
    print('Plot top-1 mean metrics.')
    print('-' * 80)
    for ligand in ['protac', 'e3', 'poi', 'linker']:
        df = scores_df.copy()

        # Get all columns that start with "default_pred_n"
        columns = [col for col in df.columns if col.startswith("default_pred_n")]

        df = df[columns]

        # Remove the "default_pred_n0_" from the column names
        df.columns = df.columns.str.replace("default_pred_n0_", "")

        # Filter columns with "heavy_atoms_difference" and "num_fragments" and "has_three_substructures" and "is_fixed"
        # df = df[df.columns[~df.columns.str.contains("heavy_atoms_difference|num_fragments|has_three_substructures|is_fixed|graph")]]
        df = df[df.columns[~df.columns.str.contains("num_fragments|has_three_substructures|is_fixed|graph")]]

        # Remove the "default_pred_n0" column
        df = df[df.columns[~df.columns.str.contains("default_pred_n0")]]

        # Filter all columns that do not contain "e3" or "poi" or "linker"
        if ligand == 'protac':
            df = df[df.columns[~df.columns.str.contains("e3|poi|linker")]]
            title = "Top-1 Mean PROTAC Metrics"
        elif ligand == 'e3':
            df = df[df.columns[df.columns.str.contains("e3_", regex=False)]]
            title = "Top-1 Mean E3 Metrics"
        elif ligand == 'poi':
            df = df[df.columns[df.columns.str.contains("poi_", regex=False)]]
            title = "Top-1 Mean Warhead Metrics"
        elif ligand == 'linker':
            df = df[df.columns[df.columns.str.contains("linker_", regex=False)]]
            title = "Top-1 Mean Linker Metrics"

        # Title case the column names
        df.columns = df.columns.str.split("_").str.join(" ").str.title()
        # Remove the "fixed" from the column names
        df.columns = df.columns.str.replace("Fixed", "(fixed)")
        
        # Sort columns from non containing "(fixed)" first and then the ones containing "(fixed)"
        df = df.reindex(sorted(df.columns, key=lambda x: (x.split(" ")[-1] != "(fixed)", x), reverse=True), axis=1)

        # First convert boolean values to numeric (True = 1, False = 0)
        df = df.astype(float)

        # Bar-plot the values of the DataFrame
        # plt.figure(figsize=(20, 10))
        ax = sns.barplot(data=df, orient='h', errorbar='ci', color='C0')
        plt.xlabel("")
        plt.ylabel("")
        plt.title(title)
        plt.xlim(0, 1.1)
        plt.grid(axis='x', alpha=0.5)
        # For each bar in the plot, add the value at x=0.5, use ax.patches to get the bars
        for p in ax.patches:
            ax.annotate(f"{p.get_width():.4f}", (0.05, p.get_y() + p.get_height() / 2),
                        ha='left', va='center', fontsize=10, color='black')

        # Convert the x-axis to percentage
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))
        if img_dir:
            plot_file = os.path.join(img_dir, f"plot_{ligand}_top1.pdf")
            plt.savefig(plot_file, bbox_inches='tight')
            plt.clf()
            plt.close()
            print(f"Plot saved to: {plot_file}")
        else:
            plt.show()

    if not processed_scores_df[processed_scores_df['top_n'] == 5].empty:
        print('-' * 80)
        print('Plot top-5 mean metrics.')
        print('-' * 80)
        for ligand in ['protac', 'e3', 'poi', 'linker']:
            df = processed_scores_df[processed_scores_df['strategy'] == best_strategy].copy()
            df = df[df['reduce'] == 'mean']
            df = df[~df['metric'].str.contains("graph", regex=False)]
            # df = df[~df['metric'].str.contains("heavy_atoms", regex=False)]
            df = df[~df['metric'].str.contains("is_fixed", regex=False)]
            df = df[~df['metric'].str.contains("num_fragments", regex=False)]
            df = df[~df['metric'].str.contains("has_three_substructures", regex=False)]
            df = df[~df['metric'].str.contains("has_all_attachment_points", regex=False)]
            df = df[df['top_n'] == 5]
            
            if ligand == 'protac':
                df = df[~(df['metric'].str.contains("e3_") | df['metric'].str.contains("poi_") | df['metric'].str.contains("linker"))]
                title = "Top-5 Mean PROTAC Metrics"
            elif ligand == 'e3':
                df = df[df['metric'].str.contains("e3_")]
                title = "Top-5 Mean E3 Metrics"
            elif ligand == 'poi':
                df = df[df['metric'].str.contains("poi_")]
                title = "Top-5 Mean Warhead Metrics"
            elif ligand == 'linker':
                df = df[df['metric'].str.contains("linker_")]
                title = "Top-5 Mean Linker Metrics"
                
            # Title case the metric names
            df['metric'] = df['metric'].str.split("_").str.join(" ").str.title()
            # Replace the "fixed" from the metric names
            df['metric'] = df['metric'].str.replace("Fixed", "(fixed)")
            
            # Convert value column to numeric
            df['value'] = df['value'].astype(float)

            # Sort the metrics from non containing "(fixed)" first and then the ones containing "(fixed)"
            df = df.sort_values(by=['metric'], key=lambda x: x.str.contains("(fixed)", regex=False), ascending=True)

            # Bar-plot the values of the DataFrame
            # plt.figure(figsize=(20, 10))
            ax = sns.barplot(data=df, orient='h', color='C0', x='value', y='metric')
            plt.xlabel("")
            plt.ylabel("")
            plt.title(title)
            plt.xlim(0, 1.1)
            plt.grid(axis='x', alpha=0.5)
            # For each bar in the plot, add the value at x=0.5, use ax.patches to get the bars
            for p in ax.patches:
                ax.annotate(f"{p.get_width():.4f}", (0.05, p.get_y() + p.get_height() / 2),
                            ha='left', va='center', fontsize=10, color='black')

            # Convert the x-axis to percentage
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))
            if img_dir:
                plot_file = os.path.join(img_dir, f"plot_{ligand}_top5.pdf")
                plt.savefig(plot_file, bbox_inches='tight')
                plt.clf()
                plt.close()
                print(f"Plot saved to: {plot_file}")
            else:
                plt.show()


    if 'default_prob_n0' in scores_df.columns:
        print('-' * 80)
        print('Perplexity scores.')
        print('-' * 80)
        for x_metric in ['default_prob_n0', 'default_perplexity_n0']:
            for hue_metric in ['default_pred_n0_reassembly', 'default_pred_n0_all_ligands_equal']:
                plt.figure(figsize=(10, 5))
                sns.histplot(data=scores_df, x=x_metric, hue=hue_metric, kde=True)

                x_label = "Perplexity" if 'perplexity' in x_metric else "Probability"
                hue_label = "Reassembly" if 'reassembly' in hue_metric else "All Ligands Equal"
                plt.xlabel(x_label)
                plt.ylabel("Frequency")
                plt.title(f"Histogram of {x_label} vs {hue_label}")
                plt.grid(axis='both', alpha=0.5)
                # Change the legend title to hue_label
                if img_dir:
                    plot_file = os.path.join(img_dir, f"plot_{x_metric}_vs_{hue_metric}.pdf")
                    plt.savefig(plot_file, bbox_inches='tight')
                    print(f"Plot saved to: {plot_file}")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()

                plt.figure(figsize=(10, 5))
                sns.histplot(data=scores_df, x=x_metric, hue=hue_metric, kde=True)
                plt.xlabel(x_label)
                plt.ylabel("Frequency")
                plt.title(f"Histogram of {x_label} vs {hue_label}")
                plt.grid(axis='both', alpha=0.5)
                plt.ylim(0, 300)
                if img_dir:
                    plot_file = os.path.join(img_dir, f"plot_{x_metric}_vs_{hue_metric}_zoom.pdf")
                    plt.savefig(plot_file, bbox_inches='tight')
                    plt.clf()
                    plt.close()
                    print(f"Plot saved to: {plot_file}")
                else:
                    plt.show()

                plt.figure(figsize=(10, 5))
                sns.scatterplot(data=scores_df, x=x_metric, y=hue_metric)
                plt.xlabel(x_label)
                plt.ylabel("Frequency")
                plt.title(f"Scatter of {x_label} vs {hue_label}")
                plt.grid(axis='both', alpha=0.5)
                if img_dir:
                    plot_file = os.path.join(img_dir, f"plot_{x_metric}_vs_{hue_metric}_scatter.pdf")
                    plt.savefig(plot_file, bbox_inches='tight')
                    plt.clf()
                    plt.close()
                    print(f"Plot saved to: {plot_file}")
                else:
                    plt.show()

        # Fit a logistic regression model to predict the threshold value for the "default_prob_n0" column
        # Split the data into training and testing sets
        X = scores_df['default_prob_n0'].values.reshape(-1, 1)
        y = scores_df['default_pred_n0_reassembly']
        # Balance the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Oversample the minority class
        # X_train, y_train = SMOTE(k_neighbors=5, random_state=42).fit_resample(X_train, y_train)
        # X_train, y_train = ADASYN(n_neighbors=5, random_state=42).fit_resample(X_train, y_train)

        # Print the number of samples per class of the y train and test sets
        print(f"Train samples per class: {np.bincount(y_train)}")
        print(f"Test samples per class: {np.bincount(y_test)}")

        # Plot the True/False values of the y train and test sets
        plt.figure(figsize=(10, 5))
        sns.histplot(y_train, color='C0', label='Train', kde=True)
        sns.histplot(y_test, color='C1', label='Test', kde=True)
        plt.xlabel("Reassembly")
        plt.ylabel("Frequency")
        plt.title("Histogram of Reassembly Prediction")
        plt.grid(axis='both', alpha=0.5)
        plt.legend()
        if img_dir:
            plot_file = os.path.join(img_dir, f"plot_reassembly_prediction_histogram.pdf")
            plt.savefig(plot_file, bbox_inches='tight')
            plt.clf()
            plt.close()
            print(f"Plot saved to: {plot_file}")
        else:
            plt.show()

        # Fit the logistic regression model
        model = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000, tol=1e-7, Cs=100)
        model.fit(X_train, y_train)

        # Get the model score
        accuracy = model.score(X_test, y_test)
        f1score = f1_score(y_test, model.predict(X_test))
        roc_auc = roc_auc_score(y_test, model.predict(X_test))

        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_train)
        dummy_accuracy = dummy.score(X_test, y_test)
        dummy_f1score = f1_score(y_test, dummy.predict(X_test))
        dummy_roc_auc = roc_auc_score(y_test, dummy.predict(X_test))

        print(f"Model accuracy: {accuracy:.2f} (dummy: {dummy_accuracy:.2f})")
        print(f"Model F1-score: {f1score:.2f} (dummy: {dummy_f1score:.2f})")
        print(f"Model ROC-AUC:  {roc_auc:.2f} (dummy: {dummy_roc_auc:.2f})")

        x = np.linspace(min(scores_df['default_prob_n0']), max(scores_df['default_prob_n0']), 1000)
        y = model.predict_proba(x.reshape(-1, 1))[:, 1]

        # Find the y-value closest to 0.5
        y_closest = np.abs(y - 0.5)
        idx = np.argmin(y_closest)
        x_threshold = x[idx]
        y_threshold = y[idx]

        # Get the value of X for which the model predicts a 50% probability
        threshold = -model.intercept_[0] / model.coef_[0][0]

        print(f"Threshold value: {threshold} (-intercept / coef)")
        print(f"Threshold value: {x_threshold} (at likelihood: {y_threshold:.3})")

        # Plot the logistic regression curve over the scatter points
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=scores_df, x='default_prob_n0', y='default_pred_n0_reassembly')
        plt.plot([threshold, threshold], [0, 1], 'C1--')
        plt.plot(x, y, color='C2')
        # Print the threshold value on the plot
        plt.text(threshold - threshold / 1e3, 0.5, f"Threshold: {threshold:.4f}", ha='right', va='center')

        min_pos_threshold = scores_df[scores_df['default_pred_n0_reassembly'] == 1]['default_prob_n0'].min()
        plt.plot([min_pos_threshold, min_pos_threshold], [0, 1], 'C4--')
        plt.text(min_pos_threshold - min_pos_threshold / 1e2, 0.5, f"Min Positive: {min_pos_threshold:.4f}", ha='left', va='center')

        plt.xlabel("Likelihoods of PROTAC-Splitter Predictions")
        plt.ylabel("Reassembly")
        plt.title("")
        plt.grid(axis='both', alpha=0.5)
        if img_dir:
            plot_file = os.path.join(img_dir, f"plot_confidence_thresholds.pdf")
            plt.savefig(plot_file, bbox_inches='tight')
            plt.clf()
            plt.close()
            print(f"Plot saved to: {plot_file}")
        else:
            plt.show()


    def split_row(row):
        ligands = split_prediction(row['default_pred_n0'])
        label_ligands = split_prediction(row['label_smiles'])
        
        ret = row.to_dict()
        ret['e3'] = ligands['e3']
        ret['linker'] = ligands['linker']
        ret['poi'] = ligands['poi']
        ret['e3_label'] = label_ligands['e3']
        ret['linker_label'] = label_ligands['linker']
        ret['poi_label'] = label_ligands['poi']
        return ret

    if img_dir is None:
        print("No image directory supplied. Skipping image generation.")
        return

    # Get the failed reassembly predictions
    failed_reassembly_df = scores_df[~scores_df[failed_metric_to_display]]
    tqdm.pandas(desc="Splitting PROTACs into its ligands")
    failed_reassembly_df = failed_reassembly_df.progress_apply(split_row, axis=1, result_type='expand')

    for i, row in tqdm(failed_reassembly_df.sample(n=len(failed_reassembly_df), random_state=42).iterrows(), desc="Saving images to SVG", total=len(failed_reassembly_df)):    
        protac_smiles = row['protac_smiles']
        ligands_smiles = row['label_smiles']
        pred_smiles = row['default_pred_n0']
        
        ligands = split_prediction(ligands_smiles)

        if print_failed_predictions:
            print(f"{i:5d} PROTAC:  {protac_smiles}")
            print(f"{i:5d} Ligands: {ligands_smiles}")
            print(f"{i:5d} Pred:    {pred_smiles}")

        svg = get_mapped_protac_img(
            protac_smiles,
            e3_smiles=ligands['e3'],
            linker_smiles=ligands['linker'],
            poi_smiles=ligands['poi'],
            w=1000,
            h=600,
            legend="",
            useSVG=True,
            display_image=False,
        )
        output_file = os.path.join(img_dir, f"image_{failed_metric_to_display}_n{i}.svg")
        if svg:
            with open(output_file, "w") as file:
                file.write(svg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting script")
    parser.add_argument(
        "--score_file",
        type=str,
        default=None,
        help="Path to the score file",
    )
    parser.add_argument(
        "--best_strategy",
        type=str,
        default="beam_search_decoding",
        help="Best generation strategy to plot for top-5 metrics",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="Directory to save images. If not supplied, image generation will be skipped.",
    )
    parser.add_argument(
        "--failed_metric_to_display",
        type=str,
        default="default_pred_n0_reassembly_fixed",
        help="Metric to display for failed reassembly",
    )
    parser.add_argument(
        "--print_failed_predictions",
        action="store_true",
        help="Print failed predictions",
    )

    args = parser.parse_args()

    main(
        score_file=args.score_file,
        best_strategy=args.best_strategy,
        failed_metric_to_display=args.failed_metric_to_display,
        img_dir=args.img_dir,
        print_failed_predictions=args.print_failed_predictions,
    )