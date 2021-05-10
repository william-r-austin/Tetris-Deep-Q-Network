"""
Created on Apr 30, 2021

@author: William, Matthew
"""
import argparse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os

ROOT_DIR = "C:/Users/William/eclipse-workspace-cs747/Tetris-Deep-Q-Network/output/dqn_v6"

GRAPHS_DIR = "C:/Users/William/Documents/GMU Grad School/10 - Spring 2021 - CS 747/Semester Project/Graphs"

ALL_KEYS = ["Apr29_235535", "Apr30_164101", "May01_183634", "May02_084221", "May02_202559", 
            "May03_095104", "May04_002529", "May04_041753", "May04_232714", "May07_053338"]

KEY_RANK_VALS = {"Apr29_235535": 4, "Apr30_164101": 10, "May01_183634": 3, "May02_084221": 5, "May02_202559": 6, 
            "May03_095104": 1, "May04_002529": 7, "May04_041753": 8, "May04_232714": 2, "May07_053338": 9}

MAX_TETROMINOS = {}

MAX_TETROMINO_EPISODE = {}

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Tetris""")

    # Tetris Parameters
    parser.add_argument("--data_path", type=str, help="Path to data file")
    return parser.parse_args()

def get_file_for_key(run_key):
    dir_list = os.listdir(ROOT_DIR)
    for dir_entry in dir_list:
        if dir_entry.startswith(run_key):
            logs_dir = os.path.join(ROOT_DIR, dir_entry, "logs")
            run_files = os.listdir(logs_dir)
            for run_file in run_files:
                if run_file.endswith("episodes.csv"):
                    episodes_file = os.path.join(logs_dir, run_file)
                    return episodes_file
        
    return None 

def remove_legend(ax_thing):
    #for ax in ax_thing:
    ax_thing.get_legend().remove()

def create_graphs(run_key):
    #args = get_args()
    #data_filename = args.data_path
    local_graph_dir = os.path.join(GRAPHS_DIR, run_key)
    os.mkdir(local_graph_dir) 
    
    data_filename = get_file_for_key(run_key)
    
    data = pd.read_csv(data_filename)

    #print(data.shape)
    #print(data.columns)

    is_training_data = (data["Replay_Memory_Full"] == True)

    training_data = data[is_training_data]

    print(training_data.shape)

    score_data = training_data["Tetrominoes"]
    loss_data = training_data["Average_Loss"].astype('float32')
    cleared_line_data = training_data["Cleared_Lines"]
    reward_data = training_data["Reward_Sum"]
    action_count_data = training_data["Action_Count"]
    

    score_data_agg = pd.DataFrame()
    score_data_agg["Average Log Loss"] = np.log(loss_data.groupby(np.arange(len(score_data)) // 1000).mean())
    score_data_agg["Model Policy"] = score_data.groupby(np.arange(len(score_data)) // 1000).mean()
    score_data_agg["Game Number"] = np.arange(len(score_data_agg)) * 1000
    score_data_agg["Random Policy"] = [12.6] * len(score_data_agg)
    score_data_agg["Cleared Lines"] = cleared_line_data.groupby(np.arange(len(cleared_line_data)) // 1000).mean()
    score_data_agg["Reward Sum"] = reward_data.groupby(np.arange(len(reward_data)) // 1000).mean()
    score_data_agg["Action Count"] = action_count_data.groupby(np.arange(len(action_count_data)) // 1000).mean()
    score_data_agg["Efficiency"] = score_data_agg["Model Policy"] / score_data_agg["Action Count"]
    
    MAX_TETROMINOS[run_key] = score_data_agg["Model Policy"].max()
    MAX_TETROMINO_EPISODE[run_key] = score_data_agg["Model Policy"].idxmax()

    #print(score_data_agg)

    seaborn_data = pd.melt(score_data_agg, id_vars=["Game Number"], value_vars=['Random Policy', 'Model Policy'],
                           var_name='Policy', value_name='Tetromino Count')
    
    cleared_line_data = pd.melt(score_data_agg, id_vars=["Game Number"], value_vars=['Cleared Lines'],
                                var_name='Policy', value_name='Cleared Lines')
    
    reward_data = pd.melt(score_data_agg, id_vars=["Game Number"], value_vars=['Reward Sum'],
                                var_name='Policy', value_name='Reward Sum')

    loss_data = pd.melt(score_data_agg, id_vars=["Game Number"], value_vars=['Average Log Loss'],
                        var_name='Average Log Loss', value_name='Log Loss Value')

    action_count_data = pd.melt(score_data_agg, id_vars=["Game Number"], value_vars=['Action Count'],
                        var_name='Policy', value_name='Action Count')
    
    efficiency_data = pd.melt(score_data_agg, id_vars=["Game Number"], value_vars=['Efficiency'],
                        var_name='Policy', value_name='Efficiency')

    #print(seaborn_data.head())
    # sns.lineplot(data=score_data_agg, x="Game Number", y=["Tetromino Count", "Random Policy"])
    # sns.lineplot(x="Game Number", y="value", hue="variable", data=pd.melt(score_data_agg, ["Game Number"]))

    #sns.lineplot(data=seaborn_data, x="Game Number", y="Tetromino Count", hue="Policy")
    #sns.lineplot(data=cleared_line_data, x="Game Number", y="Cleared Lines", hue="Policy")
    #sns.lineplot(data=reward_data, x="Game Number", y="Reward Sum", hue="Policy")
    #sns.lineplot(data=loss_data, x="Game Number", y="Log Loss Value", hue="Average Log Loss")
    
    rank_string = "(Rank #" + str(KEY_RANK_VALS[run_key]) + ")"
    
    fig1, ax1 = plt.subplots()
    sns_plot1 = sns.lineplot(data=seaborn_data, x="Game Number", y="Tetromino Count", hue="Policy", ax=ax1).set_title('Tetromino Count ' + rank_string)
    g1_name = os.path.join(local_graph_dir, "1_tetromino_count.png")
    sns_plot1.figure.savefig(g1_name)

    fig2, ax2 = plt.subplots()
    #for ax in ax2: ax.legend([],[], frameon=False)
    #remove_legend(ax2)
    sns_plot2 = sns.lineplot(data=loss_data, x="Game Number", y="Log Loss Value", hue="Average Log Loss", legend=False, ax=ax2).set_title('Average Log Loss ' + rank_string)
    #sns_plot2.get_legend().remove()
    g2_name = os.path.join(local_graph_dir, "2_log_loss.png")
    sns_plot2.figure.savefig(g2_name)
    
    fig3, ax3 = plt.subplots()
    sns_plot3 = sns.lineplot(data=cleared_line_data, x="Game Number", y="Cleared Lines", hue="Policy", legend=False, ax=ax3).set_title('Cleared Lines per Game ' + rank_string)
    g3_name = os.path.join(local_graph_dir, "3_cleared_lines.png")
    sns_plot3.figure.savefig(g3_name)
    
    fig4, ax4 = plt.subplots()
    sns_plot4 = sns.lineplot(data=reward_data, x="Game Number", y="Reward Sum", hue="Policy", legend=False, ax=ax4).set_title('Total Reward Sum ' + rank_string)
    g4_name = os.path.join(local_graph_dir, "4_reward_sum.png")
    sns_plot4.figure.savefig(g4_name)
    
    fig5, ax5 = plt.subplots()
    sns_plot5 = sns.lineplot(data=action_count_data, x="Game Number", y="Action Count", hue="Policy", legend=False, ax=ax5).set_title('Action Count ' + rank_string)
    g5_name = os.path.join(local_graph_dir, "5_action_count.png")
    sns_plot5.figure.savefig(g5_name)
    
    fig6, ax6 = plt.subplots()
    sns_plot6 = sns.lineplot(data=efficiency_data, x="Game Number", y="Efficiency", hue="Policy", legend=False, ax=ax6).set_title('Efficiency = Tetrominos/Actions ' + rank_string)
    g6_name = os.path.join(local_graph_dir, "6_efficiency.png")
    sns_plot6.figure.savefig(g6_name)

    #plt.tight_layout()
    #plt.show()

    # print("Cols = " + str(data.head()))
    
if __name__ == '__main__':
    matplotlib.style.use('seaborn-whitegrid')
    
    i = 1
    for run_key in ALL_KEYS:
        create_graphs(run_key)
        print("Done creating graph for key: " + run_key + "  (#" + str(i) + ")")
        i += 1
    
    max_scores = sorted(MAX_TETROMINOS.items(), key=lambda x: x[1])
    
    for g in reversed(max_scores):
        print(g)
    
    print("----------------")
    
    print(MAX_TETROMINO_EPISODE)