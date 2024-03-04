#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:40:33 2024

@author: nilsl
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_columns(row):
    return f"{row['study_item']}_{row['metric']}_{row.name.split('_')[0]}"

def combine_interview_parts_corrected(df):
    # Copy of the DataFrame to avoid modifying the original during iteration
    df_copy = df.copy()
    
    # Iterating over unique study_ids and metrics
    for study_id in df['study_id'].unique():
        for metric in df["metric"].unique():
            # Filtering the DataFrame for the current study_id and metric
            df_filtered = df_copy[(df_copy['study_id'] == study_id) & (df_copy["metric"] == metric)]

            # Identifying unique base names for study items (excluding part1/part2 distinction)
            base_items = df_filtered['study_item'].str.replace('_teil[12]', '', regex=True).unique()

            for base_item in base_items:
                # Find rows corresponding to part 1 and part 2 for the current base item
                part1_rows = df_filtered[df_filtered['study_item'] == f'{base_item}_teil']
                part2_rows = df_filtered[df_filtered['study_item'] == f'{base_item}_teil2']

                # Proceed if both parts exist
                if not part1_rows.empty and not part2_rows.empty:
                    # Assuming 'values' are stored as lists or arrays; adjust as needed
                    for part1_index, part1_row in part1_rows.iterrows():
                        part2_row = part2_rows[part2_rows['study_item'].str.startswith(base_item)].iloc[0]
                        combined_values = part1_row['values'] + part2_row['values']

                        # Update DataFrame with combined values and remove part 2 row
                        df_copy.at[part1_index, 'values'] = combined_values
                        df_copy = df_copy.drop(part2_row.name)

                        # Update the study_item name to remove part distinction
                        df_copy.at[part1_index, 'study_item'] = base_item

    return df_copy

def plot_density(dataframe, subplot_titles, figure_title):
    """
    Plots a density plot for each aggregated value in the dataframe with a layout close to square.

    :param dataframe: The aggregated DataFrame.
    :param subplot_titles: A list of titles for each subplot.
    :param figure_title: The overall title of the figure.
    """
    
    for j, metric in enumerate(dataframe['metric'].unique()):
        
        df = dataframe[dataframe["metric"] == metric]
    
        num_plots = len(df)
        # Calculate the number of rows and columns to make the layout as square as possible
        cols = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / cols))
    
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        fig.suptitle(f"{figure_title} Metric: {metric}")
    
        # Flatten the axes array for easy indexing
        axes_flat = axes.flatten()
    
        # Plotting each density plot
        for i in range(num_plots):
            sns.kdeplot(data=df.iloc[i]['values'], ax=axes_flat[i], fill=True)
            axes_flat[i].set_title(f"{subplot_titles[i*5+j]}")
            axes_flat[i].set_xlabel('Values')
            axes_flat[i].set_ylabel('Density')
    
        # Hide unused subplots
        for i in range(num_plots, len(axes_flat)):
            axes_flat[i].set_visible(False)
    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == '__main__':

    n = np.inf
    remove_stopwords = False
    remove_numbers = False
    
    df_metrics = combine_interview_parts_corrected(pd.read_pickle(f"velas_subsample_{n}_stopwords_{remove_stopwords}_numbers_{remove_numbers}.pkl"))
    df_metrics = df_metrics[df_metrics["metric"]== "conditional_entropy_error"]
    
    df_metrics['mean_val'] = df_metrics['values'].apply(lambda x: np.nanmean(x))
    df_metrics['sd_val'] = df_metrics['values'].apply(lambda x: np.nanstd(x))
    # df_metrics['len'] = df_metrics['values'].apply(lambda x: len(x))
    
    # # # Median
    # df_metrics['median_val'] = df_metrics['values'].apply(lambda x: np.nanmedian(x))
    
    # # # Interquartile Range (IQR)
    # # df_metrics['iqr_val'] = df_metrics['values'].apply(lambda x: np.subtract(*np.nanpercentile(x, [75, 25])))
    # from scipy.stats import skew, kurtosis, mode
    # # # Skewness
    # df_metrics['skewness_val'] = df_metrics['values'].apply(lambda x: skew(x, nan_policy='omit'))
    
    # # Kurtosis
    # df_metrics['kurtosis_val'] = df_metrics['values'].apply(lambda x: kurtosis(x, nan_policy='omit'))
    
    # # # Range
    # df_metrics['range_val'] = df_metrics['values'].apply(lambda x: np.nanmax(x) - np.nanmin(x))

    # # Quantiles - Since you need individual values, we'll create separate columns for each
    # # 25th Percentile
    # df_metrics['25th_percentile'] = df_metrics['values'].apply(lambda x: np.nanpercentile(x, 25))
    
    # # 50th Percentile (Median) - Already computed, but included for clarity if needed separately
    # df_metrics['50th_percentile'] = df_metrics['median_val']
    
    # # 75th Percentile
    # df_metrics['75th_percentile'] = df_metrics['values'].apply(lambda x: np.nanpercentile(x, 75))    


    df_metrics.loc[df_metrics["study_item"] == "panss..rtf", "study_item"] = "panss"
    df_metrics.loc[df_metrics["study_item"] == "panss.rtf", "study_item"] = "panss"
    df_metrics.loc[df_metrics["study_item"] == "panss.wav.rtf", "study_item"] = "panss"
    df_metrics.loc[df_metrics["study_item"] == "panss.wav", "study_item"] = "panss"
    
    df_metrics.loc[df_metrics["study_item"] == "discourse.rtf", "study_item"] = "discourse"
    df_metrics.loc[df_metrics["study_item"] == "disocurse.rtf", "study_item"] = "discourse"
    df_metrics.loc[df_metrics["study_item"] == "discourse_teil1", "study_item"] = "discourse"
    
    
    df_metrics.loc[df_metrics["study_item"] == "interview.wav", "study_item"] = "interview"
    df_metrics.loc[df_metrics["study_item"] == "interview_teil2", "study_item"] = "interview"
    df_metrics.loc[df_metrics["study_item"] == "interview_teil1", "study_item"] = "interview"
    df_metrics.loc[df_metrics["study_item"] == "interview_teil", "study_item"] = "interview"
    
    df_values = df_metrics[["group", "study_id", "study_item", "metric", "values"]]
    
    
    # 1. Aggregate by Group
    # group_agg = df_values.groupby(['group', 'metric'])['values'].apply(lambda x: np.concatenate(x.values)).reset_index()
    
    # # 2. Aggregate by Study Item
    # study_item_agg = df_values.groupby(['study_item', 'metric'])['values'].apply(lambda x: np.concatenate(x.values)).reset_index()
    
    # # 3. Aggregate by Group and Study Item
    # group_study_item_agg = df_values.groupby(['group', 'study_item', 'metric'])['values'].apply(lambda x: np.concatenate(x.values)).reset_index()

    
    
    # # Example usage with the group_agg DataFrame
    
    # group_titles = group_agg.apply(lambda x: f"{x['group']}", axis=1)
    # plot_density(group_agg, group_titles, 'Density Plots by Group')
    
    
    # study_item_titles = study_item_agg.apply(lambda x: f"{x['study_item']}", axis=1)
    # # Example usage with the study_item_agg DataFrame
    # plot_density(study_item_agg, study_item_titles, 'Density Plots by Study Item')
    
    # # For group_study_item_agg, creating titles that combine group and study_item
    # group_study_item_titles = group_study_item_agg.apply(lambda x: f"{x['group']} - {x['study_item']}", axis=1)
    # plot_density(group_study_item_agg, group_study_item_titles, 'Density Plots by Group and Study Item')
    
    



    
    df_metrics.drop("values", inplace = True, axis = 1)
   
        

    # Apply aggregations and drop 'values'
    df_pivoted = df_metrics.set_index(["group", "study_id", "study_item", "metric"]).unstack([-2, -1]).sort_index(axis=1, level=0)
    
    # Flatten the MultiIndex columns and create new column names
    df_pivoted.columns = ['_'.join(col).strip() for col in df_pivoted.columns.values]
    
    # df_pivoted.drop(['len_interview_conditional_entropy_error','len_panss_conditional_entropy_error','len_discourse_conditional_entropy_error','len_discourse_max_probability_difference', 'len_discourse_probability','len_discourse_similarity',
    # 'len_interview_max_probability_difference', 'len_interview_probability', 'len_interview_similarity', 
    # 'len_panss_max_probability_difference', 'len_panss_probability', 'len_panss_similarity'], axis = 1, inplace = True)
    
    df_pivoted.reset_index(inplace=True)

       
    imputer = SimpleImputer(strategy='median')
    df_imputed = df_pivoted.copy()
    df_imputed.iloc[:, 2:] = imputer.fit_transform(df_pivoted.iloc[:, 2:])  # Skip 'group', 'study_id' for imputation
    
    X = df_imputed.iloc[:, 2:]  # Feature matrix
    y = df_imputed['group']  # Target variable
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['group'] = y.values  # Add group labels to the PCA dataframe
       
    sns.scatterplot(x='PC1', y='PC2', hue='group', data=df_pca, palette='viridis')
    
    plt.title('PCA of Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Group')
    plt.show()
       
    sns.scatterplot(x='PC1', y='PC3', hue='group', data=df_pca, palette='viridis')
    
    plt.title('PCA of Dataset 2')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 3')
    plt.legend(title='Group')
    plt.show()
    
    sns.scatterplot(x='PC2', y='PC3', hue='group', data=df_pca, palette='viridis')
       
    plt.title('PCA of Dataset 3')
    plt.xlabel('Principal Component 2')
    plt.ylabel('Principal Component 3')
    plt.legend(title='Group')
    plt.show()
    
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    
    # Assuming X and y are your features and target variable
    clf = RandomForestClassifier(random_state=42, n_estimators = 100)
    
    # Define a StratifiedKFold instance
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    
    # Perform stratified 5-fold cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=stratified_cv)
    
    # Calculate the average and standard deviation of the cross-validation scores
    cv_scores_mean = np.mean(cv_scores)
    cv_scores_std = np.std(cv_scores)
    
    print(f"Average 5-Fold Stratified CV Score: {cv_scores_mean}, with standard deviation: {cv_scores_std}")
    
    # Optionally, fit the model on the entire dataset to retrieve feature importances
    clf.fit(X, y)
    feature_importances = clf.feature_importances_
    print(feature_importances)
