"""
Created on Tue Nov 28 15:10:03 2023
@author: mcanela
"""

import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

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


def process_syllables(syllables):
    # Combine specific groups of syllables and delete others
    processed_syllables = []

    for syllable in syllables:
        # if syllable in ['3', '9', '20', '23']:
        #     # Skip and delete syllables '3', '9', '20', '23'
            # continue
        if syllable in ['0', '1', '6', '13']:
            # Combine specified group into one syllable and delete original syllables
            processed_syllables.append('group1')
        elif syllable in ['2', '10', '11', '12', '14', '22', '28', '29', '32', '33', '34', '35', '36', '37']:
            # Combine specified group into one syllable and delete original syllables
            processed_syllables.append('group2')
        elif syllable in ['4', '7', '15', '16', '18', '21', '24']:
            # Combine specified group into one syllable and delete original syllables
            processed_syllables.append('group3')
        elif syllable in ['5', '8', '17', '19', '25', '26', '27', '30', '31', '38']:
            # Combine specified group into one syllable and delete original syllables
            processed_syllables.append('group4')
        else:
            # Keep other syllables unchanged
            processed_syllables.append(syllable)

    return processed_syllables


def filter_consecutive_appearances(syllable_list, threshold=10):
    current_syllable = syllable_list[0]
    consecutive_count = 1

    for i in range(1, len(syllable_list)):
        if syllable_list[i] == current_syllable:
            consecutive_count += 1
        else:
            if consecutive_count < threshold:
                syllable_list[i - consecutive_count:i] = ['out'] * consecutive_count
            current_syllable = syllable_list[i]
            consecutive_count = 1

    # Handle the last group if it is less than the threshold
    if consecutive_count < threshold:
        syllable_list[-consecutive_count:] = ['out'] * consecutive_count

    return syllable_list


def create_transition_matrix(syllables_dict):
    # Create a set of all unique states in the syllables_dict
    unique_states = set(value for values in syllables_dict.values() for value in values)
    unique_states = sorted(list(unique_states))
    unique_states = list(map(str, sorted(map(int, unique_states))))

    # Create an empty transition matrix filled with zeros
    transition_matrix = np.zeros((len(unique_states), len(unique_states)))

    # Iterate over each state's list of values and update the matrix
    for values in syllables_dict.values():
        for i in range(len(values) - 1):
            current_state_index = unique_states.index(values[i])
            next_state_index = unique_states.index(values[i + 1])
            transition_matrix[current_state_index][next_state_index] += 1

    # Normalize the matrix to get transition probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = transition_matrix / row_sums

    return transition_matrix, unique_states


def list_of_frames(total_bins, bin_num, specific): # bin_num starting from zero

    # Upload target (aka, conditions or group)
    txt_path = os.path.join(directory, 'target.txt')
    target_values = pd.read_csv(txt_path)
    # target_values['target'] = target_values.learning + '_' + target_values.group + '_' + target_values.batch
    target_values['target'] = target_values.learning + '_' + target_values.group
    target_values = target_values[target_values.target == specific]
    target_values = list(target_values.Index)
    
    syllables_dict = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            tag = filename.split('DLC')[0]
            if tag in target_values:
                csv_path = os.path.join(directory, filename)
                
                syllables = get_syllable_list(csv_path, total_bins, bin_num)
                
                # Process syllables before adding to the dictionary
                # syllables = process_syllables(syllables)
                
                # Filter consecutive appereances
                syllables = filter_consecutive_appearances(syllables)
                syllables = list(filter(lambda x: x != 'out', syllables))
                syllables_dict[tag] = syllables

    transition_matrix, unique_states = create_transition_matrix(syllables_dict)
    '''
    Each row (i) represents the current state, and each column (j) represents the next state.
    Each element transition_matrix[i][j] represents the probability of transitioning from state i to state j.
    This implies a direction from state i to state j (asymmetric matrix).
    '''
   
    return transition_matrix, unique_states

# =============================================================================
# Plot functions
# =============================================================================

def plot_transition_matrix(ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    transition_matrix, unique_states = list_of_frames(total_bins=6, bin_num=2, specific='Direct_Paired') # bin_num starting from zero
    
    # Create a DataFrame for better labeling of the heatmap
    sns.heatmap(transition_matrix, cmap="Blues", xticklabels=unique_states, yticklabels=unique_states)

    # Plot the heatmap
    sns.set_theme(style="whitegrid")

    # Set labels and title
    plt.title("Transition Probability Matrix between Syllables", loc = 'left', color='#636466')
    plt.xlabel("Next State", loc='left')
    plt.ylabel("Current State", loc='top')
    
    # Grey color
    ax.xaxis.label.set_color('#636466')
    ax.yaxis.label.set_color('#636466')
    ax.tick_params(axis='x', colors='#636466')
    ax.tick_params(axis='y', colors='#636466')

    return ax


def plot_transition_graph(ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    transition_matrix, unique_states = list_of_frames(total_bins=6, bin_num=2, specific='Direct_Paired') # bin_num starting from zero

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(range(len(unique_states)))

    # Add weighted edges based on transition probabilities
    for i in range(len(unique_states)):
        for j in range(len(unique_states)):
            if i != j and transition_matrix[i][j] > 0:  # Avoid self-transitions
                G.add_edge(i, j, weight=transition_matrix[i][j])

    # Get edge weights
    edge_weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_size=8, node_size=800, node_color='skyblue', font_color='black', width=edge_weights, edge_color='gray', font_weight='bold', arrowsize=10, ax=ax)

    ax.set_title("Transition Graph with Transition Rates")

    return ax
























