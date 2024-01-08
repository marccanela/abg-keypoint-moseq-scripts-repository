"""
Created on Mon Jan  8 11:48:13 2024
@author: mcanela
"""

import os
import csv
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
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


def tsne(ax=None):
    
    df = counts_df()
    
    df.info()
    df.describe()    
    
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(df.target))
    y_encoded.value_counts()
    
    std_scaler = StandardScaler()
    X_scaled = std_scaler.fit_transform(X)
    
    tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)
    X_reduced = tsne.fit_transform(X) # Without scaling
    
    return X_reduced, y, y_encoded


def plot_digits(min_distance=0.04, images=None, figsize=(13, 10)):
    
    X_reduced, y, y_encoded = tsne()
    
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[11., 11.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = plt.cm.jet
    digits = np.unique(y)
    for digit in range(len(digits)):        
        plt.scatter(X_reduced[y == digits[digit], 0], X_reduced[y == digits[digit], 1],
                    c=[cmap(float(digit) / 12)], alpha=0.5)
    plt.axis("off")
    ax = plt.gca()  # get current axes
    for index, image_coord in enumerate(X_reduced):
        closest_distance = np.linalg.norm(neighbors - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(y[index]),
                         color=cmap(float(y_encoded[index]) / 12),
                         fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"),
                                          image_coord)
                ax.add_artist(imagebox)


































