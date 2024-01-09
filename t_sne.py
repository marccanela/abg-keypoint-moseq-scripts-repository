"""
Created on Mon Jan  8 11:48:13 2024
@author: mcanela
"""

import os
import csv
import pandas as pd
import numpy as np

from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

directory = "/Users/mcanela/Downloads/keypoint_data_1/"


def get_syllable_list(csv_path, total_bins, bin_num):
    
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)   # Create a CSV reader       
        next(csv_reader)    # Skip the header row        
        syllable_list = [row[0] for row in csv_reader]    # Extract values from 1st column and convert to list
        
        bin_length = int(len(syllable_list)/total_bins)
        syllables = [syllable_list[i:i + bin_length] for i in range(0, len(syllable_list), bin_length)]
        syllables = syllables[bin_num]
        
    return syllables


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


def correlations(X):
    
    # First check if there's multicollineality
    # VIF=1 indicates no multicollinearity, VIF>5-10 suggests high multicollinearity 
    X_const = add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    print(vif_data)
    
    high_vif = vif_data.Variable[vif_data.VIF > 5][1:]
    filtered_X = X.filter(items=high_vif, axis=1)
    
    corr_matrix = filtered_X.corr()
    significant_pairs = {}
    p_val_threshold = 0.05
    corr_coeff_threshold = 0.7

    # Loop through each pair of columns in the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            
            # Calculate correlation coefficient and p-value
            corr_coeff, p_val = pearsonr(filtered_X[col1], filtered_X[col2])
            
            # Check if both the p-value and correlation coefficient meet the thresholds
            if p_val < p_val_threshold and abs(corr_coeff) > corr_coeff_threshold:
                significant_pairs[(col1, col2)] = (corr_coeff, p_val)
    
    # Create a set of unique significant pairs without directionality
    unique_significant_pairs = set()
    
    for pair in significant_pairs:
        unique_significant_pairs.add(tuple(sorted(pair)))
        
    # Create a graph using networkx
    G = nx.Graph()
    
    # Add nodes (instances) to the graph
    G.add_nodes_from(filtered_X.columns)
    
    # Add edges (significant pairs) to the graph
    for pair in unique_significant_pairs:
        G.add_edge(pair[0], pair[1])
    
    # Find connected components in the graph
    connected_components = list(nx.connected_components(G))
    
    # Combine features listed in connected_components into new features
    for idx, component in enumerate(connected_components):
        # Create a new feature by taking the mean of the features in the connected component
        new_feature_name = f"Group_{idx + 1}"
        X[new_feature_name] = X[list(component)].mean(axis=1)
        
        # Drop the old features in the connected component
        X.drop(list(component), axis=1, inplace=True)
    
    return X


def tsne(ax=None):
    
    df = counts_df()
    
    # df.info()
    # df.describe()    
    
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    # Look for significant correlations and combine features
    #X = correlations(X) 
    
        
    # List of columns to sum
    cols = [
        ['0','1','6','13'],
        ['2','10','11','12','14','22','28','29','32','33','34','35','36','37'],
        ['4','7','15','16','18','21','24'],
            ]
    
    for col in cols:
        # Create a new column with the sum
        X[str(col)] = X[col].sum(axis=1)
    
        # Delete the old columns
        X = X.drop(columns=col)
        
    # Change feature titles to integers
    X.columns = range(len(X.columns))
    
    std_scaler = StandardScaler()
    X_scaled = std_scaler.fit_transform(X)
    
    tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)
    X_reduced = tsne.fit_transform(X_scaled) 
    
    return X_reduced, y


# X_reduced, y = tsne()
def plot_digits(X_reduced, y, just_plot=['super_specific', 'mediated paired'], plot_by='target', ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
            
    df_reduced = pd.DataFrame(X_reduced, columns=['z1', 'z2'])
    df_reduced['target'] = y
    df_reduced['period'] = df_reduced.target.str.split(' ').str[0]
    df_reduced['learning'] = df_reduced.target.str.split(' ').str[1]
    df_reduced['control'] = df_reduced.target.str.split(' ').str[2]
    df_reduced['super_specific'] = df_reduced.target.str.split(': ').str[-1]    
    df_reduced = df_reduced[df_reduced[just_plot[0]] == just_plot[1]]

    blue = '#194680'
    red = '#801946'
    grey = '#636466'
    soft_grey = '#D3D3D3'
    
    color_dict = {
        'Before: direct paired': soft_grey,
        'During: direct paired': red,
        'Before: mediated paired': grey,
        'During: mediated paired': blue,
    }
    for tag in y.values:
        if tag not in color_dict.keys():
            color_dict[tag] = soft_grey
    
    labels, uniques = pd.factorize(df_reduced[plot_by])
    for label in uniques:
        subset = df_reduced[df_reduced[plot_by] == label]
        ax.scatter(subset['z1'], subset['z2'], label=label, c=color_dict[label])
    
    ax.axis("off")
    
    return ax































