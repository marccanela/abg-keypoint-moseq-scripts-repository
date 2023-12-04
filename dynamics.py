"""
Created on Tue Nov 28 15:10:03 2023
@author: mcanela
"""

import os
import csv
import community
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from statistics import mean
from scipy.stats import ttest_rel, mannwhitneyu


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
        if syllable in ['3', '9', '20', '23']:
            # Skip and delete syllables '3', '9', '20', '23'
            continue
        if int(syllable) > 38:
            # Skip and delete syllables bigger than '38'
            continue
        # if syllable in ['0', '1', '6', '13']:
        #     # Combine specified group into one syllable and delete original syllables
        #     processed_syllables.append('100')
        # elif syllable in ['2', '10', '11', '12', '14', '22', '28', '29', '32', '33', '34', '35', '36', '37']:
        #     # Combine specified group into one syllable and delete original syllables
        #     processed_syllables.append('101')
        # elif syllable in ['4', '7', '15', '16', '18', '21', '24']:
        #     # Combine specified group into one syllable and delete original syllables
        #     processed_syllables.append('102')
        # elif syllable in ['5', '8', '17', '19', '25', '26', '27', '30', '31', '38']:
        #     # Combine specified group into one syllable and delete original syllables
        #     processed_syllables.append('103')
        else:
            # Keep other syllables unchanged
            processed_syllables.append(syllable)

    return processed_syllables


def diagonal_noramlize(transition_matrix, unique_states):
    
    # Convert diagonal elements to zero
    np.fill_diagonal(transition_matrix, 0)

    # Step 1: Identify rows and columns with all zeros
    zero_rows = np.all(transition_matrix == 0, axis=1)
    zero_columns = np.all(transition_matrix == 0, axis=0)
    
    # Step 2: Get positions of zero rows and columns
    zero_rows_indices = np.where(zero_rows)[0]
    zero_columns_indices = np.where(zero_columns)[0]
    
    # Step 3: Merge and keep unique positions
    zero_positions = np.unique(np.concatenate((zero_rows_indices, zero_columns_indices)))
    
    # Step 4: Filter out elements from unique_states
    filtered_unique_states = [state for i, state in enumerate(unique_states) if i not in zero_positions]
    
    # Step 5: Delete rows and columns from transition_matrix
    transition_matrix = np.delete(transition_matrix, zero_positions, axis=0)
    transition_matrix = np.delete(transition_matrix, zero_positions, axis=1)
    
    # Step 6: Calculate row_sums and perform division
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = transition_matrix / row_sums
    
    return transition_matrix, filtered_unique_states


def create_transition_matrix(syllables_dict, min_frequency=0.005):
    
    list_of_states = []
    selected_keys = set()
    
    for key, value in syllables_dict.items():
        # Count the frequency of each state and filter out the less represented than min_frequency
        state_counts = {}
        for x in value:
            state_counts[x] = state_counts.get(x, 0) + 1
        total_sum = sum(state_counts.values())
        normalized_state_counts = {key: value / total_sum for key, value in state_counts.items()}
        list_of_states.append(normalized_state_counts)

    # Calculate mean frequency for each key
    key_frequencies = {}
    for state_counts in list_of_states:
        for key, frequency in state_counts.items():
            key_frequencies[key] = key_frequencies.get(key, []) + [frequency]
    
    # Filter out keys based on mean frequency
    for key, frequencies in key_frequencies.items():
        if mean(frequencies) >= min_frequency:
            selected_keys.add(key)
    
    # Sort the unique states
    unique_states = list(map(str, sorted(list(map(int, selected_keys)))))

    # Create an empty transition matrix filled with zeros
    transition_matrix = np.zeros((len(unique_states), len(unique_states)))

    # Iterate over each state's list of values and update the matrix
    for values in syllables_dict.values():
        for i in range(len(values) - 1):
            if values[i] in unique_states and values[i + 1] in unique_states:
                current_state_index = unique_states.index(values[i])
                next_state_index = unique_states.index(values[i + 1])
                transition_matrix[current_state_index][next_state_index] += 1

    transition_matrix, unique_states = diagonal_noramlize(transition_matrix, unique_states)

    return transition_matrix, unique_states


def create_single_transition_matrix(syllables_dict, min_frequency=0.005):
    
    individual_matrix_dict = {}
    
    for key, value in syllables_dict.items():
        
        selected_keys = set()
        
        # Count the frequency of each state and filter out the less represented than min_frequency
        state_counts = {}
        for x in value:
            state_counts[x] = state_counts.get(x, 0) + 1
        total_sum = sum(state_counts.values())
        normalized_state_counts = {key: value / total_sum for key, value in state_counts.items()}
        for syllable, frequencies in normalized_state_counts.items():
            if frequencies >= min_frequency:
                selected_keys.add(syllable)
 
        # Sort the unique states
        unique_states = list(map(str, sorted(list(map(int, selected_keys)))))

        # Create an empty transition matrix filled with zeros and iterate
        transition_matrix = np.zeros((len(unique_states), len(unique_states)))
        
        for i in range(len(value) - 1):
            if value[i] in unique_states and value[i + 1] in unique_states:
                current_state_index = unique_states.index(value[i])
                next_state_index = unique_states.index(value[i + 1])
                transition_matrix[current_state_index][next_state_index] += 1
                
        transition_matrix = diagonal_noramlize(transition_matrix)
        
        # Add the result to a dictionary
        individual_matrix_dict[key] = [unique_states, transition_matrix]

    return individual_matrix_dict


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
                
                syllables_dict[tag] = syllables

    transition_matrix, unique_states = create_transition_matrix(syllables_dict)
    # individual_matrix_dict = create_single_transition_matrix(syllables_dict)
    '''
    Each row (i) represents the current state, and each column (j) represents the next state.
    Each element transition_matrix[i][j] represents the probability of transitioning from state i to state j.
    This implies a direction from state i to state j (asymmetric matrix).
    '''
   
    return transition_matrix, unique_states


def create_transition_graph(total_bins, bin_num, specific):
    
    transition_matrix, unique_states = list_of_frames(total_bins, bin_num, specific) # bin_num starting from zero
    
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(range(len(unique_states)))

    # Add weighted edges based on transition probabilities
    for i in range(len(unique_states)):
        for j in range(len(unique_states)):
            if i != j and transition_matrix[i][j] > 0:  # Avoid self-transitions
                G.add_edge(i, j, weight=transition_matrix[i][j])

    return G, unique_states


# =============================================================================
# Plot functions
# =============================================================================


def plot_transition_matrix(ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
    # bin_num starting from zero
    transition_matrix, unique_states = list_of_frames(total_bins=6, bin_num=2, specific='Direct_No-shock')
    
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


def plot_transition_graph(bin_num, specific, contrast_color, ax=None):
    
    G, unique_states = create_transition_graph(6, bin_num, specific)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_title("Syllable Transition Graph: " + specific, color='#636466')

    # Get edge weights
    edge_weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]
    
    # Plot the graph with node size proportional to degree centrality
    degree_centrality = nx.degree_centrality(G)
    node_size = [2000 * degree_centrality[node] for node in G.nodes()]
    
    # To identify communities or groups of nodes with higher internal connections
    # compared to external connections, you can use community detection algorithms.
    # One popular algorithm for community detection is the Louvain method. 
    
    # Convert the directed graph to an undirected graph and apply the Louvain method
    G_undirected = G.to_undirected()
    partition = community.best_partition(G_undirected)
    cmap = plt.cm.get_cmap("Pastel1", max(partition.values()) + 1)
    
    # Draw the graph
    pos = nx.spring_layout(G) # It can also be nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, font_size=8, node_size=node_size, 
            node_color=list(partition.values()), cmap=cmap, edgecolors=contrast_color,
            font_color='#636466', width=edge_weights, edge_color=contrast_color, font_weight='bold', 
            arrowsize=10, ax=ax, labels={i: state for i, state in enumerate(unique_states)})

    # blue color storytelling = #194680
    # red color storytelling = #801946
    # grey color storytelling = #636466

    return ax


def graph_iterating():
    
    directory = '//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-07a08 SOC Controls_males/Keypoint analysis/figures_graph_data_1_new/'
    learnings = ['Direct', 'Mediated']
    groups = ['Paired', 'Unpaired', 'No-shock']
    bin_nums = [2, 3]
    
    for group in groups:
        for learning in learnings:
            for bin_num in bin_nums:
                if bin_num == 2:
                    color = '#636466'
                elif bin_num == 3:
                    if learning == 'Direct':
                        color = '#801946'
                    elif learning == 'Mediated':
                        color = '#194680'
                               
                specific = learning + '_' + group
                save = learning + '_' + group + '_' + str(bin_num)
                path = os.path.join(directory, f'{save}.png')
                plot_transition_graph(bin_num, specific, color, ax=None)
                plt.savefig(path)
                plt.close()


# =============================================================================
# Measures and statistics
# =============================================================================


def structural_measures(total_bins, bin_num, specific):
    
    G, unique_states = create_transition_graph(total_bins, bin_num, specific)
    results_dict = {}

    # Calculate and return node and edge counts
    node_count = G.number_of_nodes()
    results_dict["Number of Syllables"] = node_count
    edge_count = G.number_of_edges()
    results_dict["Number of Transitions"] = edge_count
       
    # # Calculate the shortest path between all pairs of nodes
    # unique_path_lengths = []
    
    # for source in G.nodes():
    #     for target in G.nodes():
    #         if source != target:
    #             try:
    #                 length = nx.shortest_path_length(G, source=source, target=target)
    #                 unique_path_lengths.append(length)
    #             except nx.NetworkXNoPath:
    #                 pass
    
    # results_dict["Average Path Length"] = unique_path_lengths
    # Calculate the average path length for the entire graph (it's the mean of the above list)
    average_path_length = nx.average_shortest_path_length(G)
    results_dict["Average Path Length"] = average_path_length

    # Calculate the degree distribution
    degree_sequence = [G.out_degree(node) for node in G.nodes()]
    results_dict["Degree Distribution"] = degree_sequence
    
    # Calculate the clustering coefficient for individual nodes
    node_clustering = nx.clustering(G)
    node_clustering_list = [value for value in node_clustering.values()]
    results_dict["Clustering Coefficient"] = node_clustering_list
    # Calculate the clustering coefficient for the entire graph (it's the mean of the above list)
    # average_clustering = nx.average_clustering(G, count_zeros=True)  # count_zeros=True includes nodes with zero clustering coefficient

    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    degree_centrality_list = [value for value in degree_centrality.values()]
    results_dict["Degree Centrality"] = degree_centrality_list
    
    betweenness_centrality = nx.betweenness_centrality(G)
    betweenness_centrality_list = [value for value in betweenness_centrality.values()]
    results_dict["Betweenness Centrality"] = betweenness_centrality_list
    
    eigenvector_centrality = nx.eigenvector_centrality(G)
    eigenvector_centrality_list = [value for value in eigenvector_centrality.values()]
    results_dict["Eigenvector Centrality"] = eigenvector_centrality_list

    return results_dict


# def plotting_single_values(measure="Number of Transitions", group='Paired', ax=None):
    
#     dictionary_before = {}
#     dictionary_during = {}
#     learnings = ['Direct', 'Mediated'] 
#     for learning in learnings:
#         specific = learning + '_' + group
        
#         data1 = structural_measures(6, 2, specific)
#         data1 = data1[measure]
#         dictionary_before[learning] = data1
    
#         data2 = structural_measures(6, 3, specific)
#         data2 = data2[measure]
#         dictionary_during[learning] = data2
    
#     df = pd.DataFrame()
#     df['clf_name'] = learnings
#     df['auc_max'] = list(dictionary_before.values())
#     df['auc_min'] = list(dictionary_during.values())


#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(6,2),)
#         sns.set(style="whitegrid")
    
#     ax.set_xlabel(measure, loc='left')
#     ax.set_ylabel('', loc='top')
#     ax.set_title('Young adult males: ' + group, loc='left', color='#636466')

        
#     blue = '#194680'
#     red = '#801946'
#     grey = '#636466'
    
#     DOT_SIZE = 120
    
#     # create the various dots
#     # avg dot
    
#     # during dot
#     ax.scatter(
#         x=df["auc_min"],
#         y=df["clf_name"],
#         s=DOT_SIZE,
#         alpha=1,
#         color=[red, blue],
#         label="During the cue",
#         edgecolors="white",
#     )
    
#     # before dot
#     ax.scatter(
#         x=df["auc_max"],
#         y=df["clf_name"],
#         s=DOT_SIZE,
#         alpha=1,
#         color=grey,
#         label="Before the cue",
#         edgecolors="white",
#     )
    
#     # create the horizontal line
#     # between min and max vals
#     ax.hlines(
#         y=df["clf_name"],
#         xmin=df["auc_min"],
#         xmax=df["auc_max"],
#         color="grey",
#         alpha=0.4,
#         lw=4, # line-width
#         zorder=0, # make sure line at back
#     )
    
#     x_min, x_max = ax.get_xlim()
#     y_min, y_max = ax.get_ylim()
    
#     # iterate through each result and apply the text
#     # df should already be sorted
#     for i in range(0, df.shape[0]):
    
#         # add thin leading lines towards classifier names
#         # to the right of max dot
#         ax.plot(
#             [df["auc_max"][i] + 0.02, 0.6],
#             [i, i],
#             linewidth=1,
#             color="grey",
#             alpha=0.4,
#             zorder=0,
#         )
        
#         # to the left of min dot
#         ax.plot(
#             [-0.05, df["auc_min"][i] - 0.02],
#             [i, i],
#             linewidth=1,
#             color="grey",
#             alpha=0.4,
#             zorder=0,
#         )
    
#     # remove the y ticks
#     ax.set_yticks([])
    
#     # Set y-axis limits and ticks manually to center labels
#     ax.set_ylim(-0.5, len(df["clf_name"]) - 0.5)
#     ax.set_yticks(range(len(df["clf_name"])))
    
#     # drop the gridlines (inherited from 'seaborn-whitegrid' style)
#     # and drop all the spines
#     # ax.grid(False)  
#     ax.spines["top"].set_visible(False)
#     # ax.spines["bottom"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["left"].set_visible(False)
    
#     ax.xaxis.label.set_color('#636466')
#     ax.yaxis.label.set_color('#636466')
#     ax.tick_params(axis='x', colors='#636466')
#     ax.tick_params(axis='y', colors='#636466')
    
#     plt.xlim(0, 500)
    
#     # Invert the X-axis
#     ax.invert_xaxis()
    
#     plt.tight_layout()
#     return ax


# def plot_measures_iterating():
    
    directory = '//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-07a08 SOC Controls_males/Keypoint analysis/figures_graph_data_1/'
    groups = ['Paired', 'Unpaired', 'No-shock']
    measures = ['Number of Syllables', 'Number of Transitions', 'Average Path Length']
    
    for measure in measures:
        for group in groups:
            save_tag = group + '_' + measure
            path = os.path.join(directory, f'{save_tag}.png')
            ax = plotting_single_values(measure, group)
            plt.savefig(path)
            plt.close()


def plotting_structural_measures(measure="Number of Syllables", specific='Direct_Paired', contrast_color='red', ax=None):
    
    data1_position = 0
    data1 = structural_measures(6, 2, specific)
    data1 = data1[measure]
    data1_mean = np.mean(data1)
    data1_error = np.std(data1, ddof=1)
    
    data2_position = 1
    data2 = structural_measures(6, 3, specific)
    data2 = data2[measure]
    data2_mean = np.mean(data2)
    data2_error = np.std(data2, ddof=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5,4))
    
    ax.hlines(data1_mean, xmin=data1_position-0.25, xmax=data1_position+0.25, color='#636466', linewidth=1.5)
    ax.hlines(data2_mean, xmin=data2_position-0.25, xmax=data2_position+0.25, color='#636466', linewidth=1.5)
    
    ax.errorbar(data1_position, data1_mean, yerr=data1_error, lolims=False, capsize = 3, ls='None', color='#636466', zorder=-1)
    ax.errorbar(data2_position, data2_mean, yerr=data2_error, lolims=False, capsize = 3, ls='None', color='#636466', zorder=-1)

    ax.set_xticks([data1_position, data2_position])
    ax.set_xticklabels(['Before the cue', 'During the cue'])
    
    jitter = 0.15 # Dots dispersion
    
    # blue color storytelling = #194680
    # red color storytelling = #801946
    # grey color storytelling = #636466
    
    dispersion_values_data1 = np.random.normal(loc=data1_position, scale=jitter, size=len(data1)).tolist()
    ax.plot(dispersion_values_data1, data1,
            'o',                            
            markerfacecolor='#636466',    
            markeredgecolor='#636466',
            markeredgewidth=1,
            markersize=5, 
            label='Data1')      
    
    dispersion_values_data2 = np.random.normal(loc=data2_position, scale=jitter, size=len(data2)).tolist()
    ax.plot(dispersion_values_data2, data2,
            'o',                          
            markerfacecolor=contrast_color,    
            markeredgecolor=contrast_color,
            markeredgewidth=1,
            markersize=5, 
            label='Data2')   
    
    # plt.ylim(0,100)
    ax.set_xlabel('')
    ax.set_ylabel(measure.capitalize(), loc='top')    
    plt.title(specific, loc = 'left', color='#636466')
    
    # Grey color
    ax.xaxis.label.set_color('#636466')
    ax.yaxis.label.set_color('#636466')
    ax.tick_params(axis='x', colors='#636466')
    ax.tick_params(axis='y', colors='#636466')
    
    plt.tight_layout()
    return ax


def plot_measures_iterating():
    
    directory = '//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-07a08 SOC Controls_males/Keypoint analysis/figures_graph_data_1_new/'
    learnings = ['Direct', 'Mediated']
    groups = ['Paired', 'Unpaired', 'No-shock']
    measures = ['Betweenness Centrality', 'Clustering Coefficient', 'Degree Centrality',
                'Degree Distribution', 'Eigenvector Centrality']
    
    for measure in measures:
        for learning in learnings:
            if learning == 'Direct':
                color = '#801946'
            elif learning == 'Mediated':
                color = '#194680'
                
            for group in groups:
                specific = learning + '_' + group
                save_tag = specific + '_' + measure
                path = os.path.join(directory, f'{save_tag}.png')
                ax = plotting_structural_measures(measure, specific, color)
                plt.savefig(path)
                plt.close()
                
                
def statistics():

    learnings = ['Direct', 'Mediated']
    groups = ['Paired', 'Unpaired', 'No-shock']
    measures = ['Betweenness Centrality', 'Clustering Coefficient', 'Degree Centrality',
                'Degree Distribution', 'Eigenvector Centrality']
    
    # Initialize a dictionary to store results
    results_dict = {'Learning': [], 'Group': [], 'Measure': [], 'T-statistic': [], 'P-value': []}
    decimal_places=4
    
    for measure in measures:
        for learning in learnings:               
            for group in groups:
                specific = learning + '_' + group

                data1 = structural_measures(6, 2, specific)
                data1 = data1[measure]
                data2 = structural_measures(6, 3, specific)
                data2 = data2[measure]

                # Initialize t_stat and p_value variables
                t_stat, p_value = None, None
    
                # Perform statistical test (e.g., paired t-test or Wilcoxon signed-rank test)
                # Use t-test for normally distributed data, and Wilcoxon test otherwise

                # Assuming normal distribution for simplicity (you may need to check this)
                # t_stat, p_value = ttest_rel(data1, data2) # Datasets must have equal lengths
                # Wilcoxon signed-rank test for non-normally distributed data
                _, p_value = mannwhitneyu(data1, data2)
    
                # Append results to the dictionary, rounding p-value to specified decimal places
                results_dict['Learning'].append(str(learning))
                results_dict['Group'].append(str(group))
                results_dict['Measure'].append(str(measure))
                results_dict['T-statistic'].append(t_stat)
                results_dict['P-value'].append(round(p_value, decimal_places))

    # Create a DataFrame from the results dictionary
    results_df = pd.DataFrame(results_dict)
    # Pivot the dataframe (the opposite of melting)
    results_df = results_df.pivot(index=['Learning', 'Group'], columns='Measure', values='P-value').reset_index()

    return results_df


def give_single_values():
    
    learnings = ['Direct', 'Mediated']
    groups = ['No-shock']
    measures = ['Number of Syllables', 'Number of Transitions']
    
    for learning in learnings:
        for measure in measures:               
            for group in groups:
                specific = learning + '_' + group
                data = structural_measures(6, 2, specific)
                data = data[measure]
                print(measure + ' - ' + specific + ' - Before: ' + str(data))
                data2 = structural_measures(6, 3, specific)
                data2 = data2[measure]
                print(measure + ' - ' + specific + ' - During: ' + str(data2))





