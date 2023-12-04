"""
Created on Mon Nov 20 10:10:03 2023
@author: mcanela
"""

import os
import csv
import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
from scipy.stats import mannwhitneyu

directory = "//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-07a08 SOC Controls_males/Keypoint analysis/keypoint_data_1/"


def get_syllable_list(csv_path, total_bins, bin_num):
    
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)   # Create a CSV reader       
        next(csv_reader)    # Skip the header row        
        syllable_list = [row[0] for row in csv_reader]    # Extract values from 1st column and convert to list
        
        bin_length = int(len(syllable_list)/total_bins)
        syllables = [syllable_list[i:i + bin_length] for i in range(0, len(syllable_list), bin_length)]
        syllables = syllables[bin_num]
        
    return syllables


def combining_syllables(df):
    
    # List of columns to sum
    cols = [
        ['0','1','6','13'],
        ['2','10','11','12','14','22','28','29','32','33','34','35','36','37'],
        ['4','7','15','16','18','21','24'],
            ]
    
    for col in cols:
        # Create a new column with the sum
        df[str(col)] = df[col].sum(axis=1)
    
        # Delete the old columns
        df = df.drop(columns=col)
    
    return df


def count_frames_syllable(total_bins, bin_num): #bin_num starting from zero
    
    syllables_dict = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            tag = filename.split('DLC')[0]
            csv_path = os.path.join(directory, filename)
            
            syllables = get_syllable_list(csv_path, total_bins, bin_num)
            syllables_dict[tag] =syllables
    
    # Get the unique numbers across all lists
    all_numbers = set(number for sublist in syllables_dict.values() for number in sublist)
    
    # Create an empty DataFrame with columns as unique numbers
    df = pd.DataFrame(columns=sorted(all_numbers))
    
    # Iterate over the dictionary items (key-value pairs)
    for key, value in syllables_dict.items():
        # Use the Counter class to count occurrences of each number in the list
        counts = pd.Series(value).value_counts().sort_index()
        
        # Add a new row to the DataFrame with the key as the index
        df.loc[key] = counts
    
    # Fill NaN values with 0 (for numbers that didn't appear in some lists)
    df = df.fillna(0).astype(int)
    
    # Sort by order all columns
    df.columns = df.columns.astype(int)
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.loc[:, 0:39] # Select only x syllables of the dendrogram
    df.columns = df.columns.astype(str)
    
    # Drop artifact columns
    # df = df.drop('3', axis=1)
    # df = df.drop('9', axis=1)
    # df = df.drop('20', axis=1)
    # df = df.drop('23', axis=1)
    
    # Combining similar syllables into one
    # df = combining_syllables(df)
    
    # Upload target (aka, conditions or group)
    txt_path = os.path.join(directory, 'target.txt')
    target_values = pd.read_csv(txt_path)
    # target_values['target'] = target_values.learning + '_' + target_values.group + '_' + target_values.batch
    target_values['target'] = target_values.learning + '_' + target_values.group
    
    # Check lengths
    if len(target_values) != len(df):
        raise ValueError(f"Length mismatch: DataFrame has {len(df)} rows, but text file has {len(target_values)} lines.")
    
    # Step 2: Merge DataFrames
    df['Index'] = df.index
    df = pd.merge(df, target_values[['Index', 'target']], how='left', left_on='Index', right_on='Index')
    df = df.drop('Index', axis=1)
    df = df.reset_index(drop=True)
    
    return df


def standarize_the_data(df):
    
    df = df.dropna(axis=1)
    features = df.columns[:-1].tolist()

    # Separating out the features
    x = df.loc[:, features].values
    
    # Separating out the target
    y = df.loc[:,['target']].values
    
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    
    return x, y


def pca_projection_2D(x, df):
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
    
    # Print the loadings (correlation between features and principal components)
    loadings = pca.components_
    features = df.columns[:-1]

    print("Top features contributing to PC1:")
    print(features[np.abs(loadings[0, :]).argsort()[::-1]])

    print("\nTop features contributing to PC2:")
    print(features[np.abs(loadings[1, :]).argsort()[::-1]])
    
    return finalDf, pca


def counts_df():
    
    df1 = count_frames_syllable(total_bins=6, bin_num=2)
    df1 = df1[df1.target == 'Direct_Paired']
    df1['target'] = 'Before: direct paired'
    
    df2 = count_frames_syllable(total_bins=6, bin_num=2)
    df2 = df2[df2.target == 'Mediated_Paired']
    df2['target'] = 'Before: mediated paired'
    
    df3 = count_frames_syllable(total_bins=6, bin_num=2)
    df3 = df3[df3.target == 'Direct_Unpaired']
    df3['target'] = 'Before: direct unpaired'
    
    df4 = count_frames_syllable(total_bins=6, bin_num=2)
    df4 = df4[df4.target == 'Mediated_Unpaired']
    df4['target'] = 'Before: mediated unpaired'
    
    df5 = count_frames_syllable(total_bins=6, bin_num=2)
    df5 = df5[df5.target == 'Direct_No-shock']
    df5['target'] = 'Before: direct no-shock'
    
    df6 = count_frames_syllable(total_bins=6, bin_num=2)
    df6 = df6[df6.target == 'Mediated_No-shock']
    df6['target'] = 'Before: mediated no-shock'
    
    #---
    
    df7 = count_frames_syllable(total_bins=6, bin_num=3)
    df7 = df7[df7.target == 'Direct_Paired']
    df7['target'] = 'During: direct paired'
    
    df8 = count_frames_syllable(total_bins=6, bin_num=3)
    df8 = df8[df8.target == 'Mediated_Paired']
    df8['target'] = 'During: mediated paired'
    
    df9 = count_frames_syllable(total_bins=6, bin_num=3)
    df9 = df9[df9.target == 'Direct_Unpaired']
    df9['target'] = 'During: direct unpaired'
    
    df10 = count_frames_syllable(total_bins=6, bin_num=3)
    df10 = df10[df10.target == 'Mediated_Unpaired']
    df10['target'] = 'During: mediated unpaired'
    
    df11 = count_frames_syllable(total_bins=6, bin_num=3)
    df11 = df11[df11.target == 'Direct_No-shock']
    df11['target'] = 'During: direct no-shock'
    
    df12 = count_frames_syllable(total_bins=6, bin_num=3)
    df12 = df12[df12.target == 'Mediated_No-shock']
    df12['target'] = 'During: mediated no-shock'
    
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12], axis=0)
    df = df.reset_index()
    df = df.drop('index', axis=1)
    # Reorder columns to move 'target' to the end
    columns = df.columns.tolist()
    columns.remove('target')
    columns.append('target')
    df = df[columns]
    
    return df


def statistics_on_the_enrichment(df, numeric_columns, groups):
    
    # Perform Mann-Whitney U test for each pair of columns
    results = []
    
    for col in numeric_columns.columns:
        group1 = df[df['target'] == groups[0]][col]
        group2 = df[df['target'] == groups[1]][col]
    
        stat, p_value = mannwhitneyu(group1, group2)
        results.append({'Variable': col, 'Statistic': stat, 'P-Value': round(p_value, 3)})
    
    results_df = pd.DataFrame(results)
    
    # Filter significant syllables (p-value less than 0.05)
    significant_syllables = results_df[results_df['P-Value'] < 0.05]['Variable'].tolist()
    
    # Categorize significant syllables based on increase or decrease
    increased_syllables = []
    decreased_syllables = []
    
    for syllable in significant_syllables:
        group1_mean = df[df['target'] == groups[0]][syllable].mean()
        group2_mean = df[df['target'] == groups[1]][syllable].mean()
    
        if group2_mean > group1_mean:
            increased_syllables.append(syllable)
        else:
            decreased_syllables.append(syllable)
    
    significant = increased_syllables + decreased_syllables

    return results_df, significant, increased_syllables, decreased_syllables


def enrichment_plot(ax=None):
    
    df = counts_df()
    targets = [
        # 'Before: direct paired', 'During: direct paired',
        # 'Before: mediated paired', 'During: mediated paired',
        # 'Before: direct unpaired', 'During: direct unpaired',
        # 'Before: mediated unpaired', 'During: mediated unpaired',
        # 'Before: direct no-shock', 'During: direct no-shock',
        'Before: mediated no-shock', 'During: mediated no-shock',
        ]
    df = df[df.target.isin(targets)]
    groups = df['target'].unique()
    numeric_columns = df.select_dtypes(include='number')
    
    # Reshape the DataFrame to have a separate column for each numeric column and each group
    reshaped_df = pd.DataFrame(columns=['Variable', 'Group', 'Value'])
    
    for col in numeric_columns.columns:
        for group in groups:
            subset = df[df['target'] == group][col]
            reshaped_df = pd.concat([reshaped_df, pd.DataFrame({'Variable': [col] * len(subset),
                                                                'Group': [group] * len(subset),
                                                                'Value': subset.values})])
            
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
        
    # blue color storytelling = #194680
    # red color storytelling = #801946
    # grey color storytelling = #636466
    custom_palette = ['#636466', '#194680']
    sns.set(style="whitegrid")
    sns.set_palette(custom_palette)
    
    # Create the bar plot
    sns.barplot(x='Variable', y='Value', hue='Group', data=reshaped_df)
    
    # Perform a statistics on the enrichment
    results_df, significant, increased_syllables, decreased_syllables = statistics_on_the_enrichment(df, numeric_columns, groups)
    
    # Add significance stars and adjust bar heights
    for col in numeric_columns.columns:
        if col in significant:
            group1_mean = df[df['target'] == groups[0]][col].mean()
            group2_mean = df[df['target'] == groups[1]][col].mean()
            p_value = results_df[results_df['Variable'] == col]['P-Value'].values[0]
            height = max(group1_mean, group2_mean) + 50  # Adjust as needed
            
            # Add significance stars above the bars with stars based on p-value
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'

            ax.text(col, height, significance, fontsize=12, color='black', ha='center')
    
    # Other styling and formatting
    plt.legend(title='')
    leg = plt.legend()
    for text in leg.get_texts():
        text.set_color('#636466')
    
    ax.xaxis.label.set_color('#636466')
    ax.yaxis.label.set_color('#636466')
    ax.tick_params(axis='x', colors='#636466')
    ax.tick_params(axis='y', colors='#636466')
    
    ax.set_xlabel('Syllables', loc='left')
    ax.set_ylabel('Number of appearances', loc='top')
    ax.set_title('Syllable enrichment of Unsupervised Analysis: Probetest', loc='left', color='#636466')
    
    plt.ylim(0, 1200)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


def plot_pca(ax=None):
    
    df = counts_df()
    
    x, y = standarize_the_data(df)
    finalDf, pca = pca_projection_2D(x, df)
    pc1, pc2 = pca.explained_variance_ratio_
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
        
    title = 'PCA of Unsupervised Analysis: Probetest'
    targets = ['Before: direct paired', 'Before: mediated paired',
               'Before: direct unpaired', 'Before: mediated unpaired',
               'Before: direct no-shock', 'Before: mediated no-shock',
               'During: direct paired', 'During: mediated paired',
               'During: direct unpaired', 'During: mediated unpaired',
               'During: direct no-shock', 'During: mediated no-shock']
    colors = ['#FFFFFF00', '#FFFFFF00',
              '#FFFFFF00', '#FFFFFF00',
              '#FFFFFF00', '#636466',
              '#FFFFFF00', '#FFFFFF00',
              '#FFFFFF00', '#FFFFFF00',
              '#FFFFFF00', '#194680']
    
    # blue color storytelling = #194680
    # red color storytelling = #801946
    # grey color storytelling = #636466
    
    sns.set_theme(style="whitegrid")
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    
    # Grey color
    ax.xaxis.label.set_color('#636466')
    ax.yaxis.label.set_color('#636466')
    ax.tick_params(axis='x', colors='#636466')
    ax.tick_params(axis='y', colors='#636466')

    ax.set_xlabel('PC1 (EV: ' + str(pc1)[:4] + ')', loc='left')
    ax.set_ylabel('PC2 (EV: ' + str(pc2)[:4] + ')', loc='top')
    ax.set_title(title, loc = 'left', color='#636466')
    
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    distributions = []  # List to store the distributions for Wasserstein distance calculation
    

    for target, color in zip(targets,colors):
        
        if color == '#FFFFFF00':
            continue  # Skip transparent colors
        
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
        
        # Calculate the centroid
        centroid = finalDf.loc[indicesToKeep, ['principal component 1', 'principal component 2']].mean()
        
        # Calculate the covariance matrix for the group
        cov_matrix = np.cov(finalDf.loc[indicesToKeep, ['principal component 1', 'principal component 2']].T)
        
        # Determine alpha based on transparency
        alpha = 0.1 if color == '#FFFFFF00' else 0.3
        
        # Plot ellipse around the centroid using the covariance matrix
        ellipse = Ellipse(xy=centroid, width=np.sqrt(cov_matrix[0, 0])*2, height=np.sqrt(cov_matrix[1, 1])*2,
                          edgecolor=color, facecolor=color, alpha=alpha, linewidth=1)
        ax.add_patch(ellipse)
        
        ellipse = Ellipse(xy=centroid, width=np.sqrt(cov_matrix[0, 0])*3, height=np.sqrt(cov_matrix[1, 1])*3,
                          edgecolor=color, facecolor=color, alpha=alpha, linewidth=1)
        ax.add_patch(ellipse)  
        
        # Add a tag right next to the ellipse
        tag_x = centroid[0] + np.sqrt(cov_matrix[0, 0])*1.5  # Adjust the x-coordinate for the tag
        tag_y = centroid[1] + np.sqrt(cov_matrix[1, 1])*1.5  # Adjust the y-coordinate for the tag
        ax.text(tag_x, tag_y, target, color=color, fontsize=12, weight='bold')
        

        # Store the data points for Wasserstein distance calculation
        distribution = finalDf.loc[indicesToKeep, ['principal component 1', 'principal component 2']]
        distributions.append(distribution.values)
    
    # Calculate Wasserstein distance
    wasserstein_dist = wasserstein_distance(distributions[0].ravel(), distributions[1].ravel())
    wasserstein_text = 'Wasserstein distance = ' + str(wasserstein_dist)[:4]
    ax.text(-5, -5, wasserstein_text, color='#636466', fontsize=11)
    
    ax.grid()
    
    plt.tight_layout()
    return ax
    
    
# plot_pca(ax=None)




















