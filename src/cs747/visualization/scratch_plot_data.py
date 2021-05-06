"""
Created on Apr 30, 2021

@author: William, Matthew
"""
import argparse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Tetris""")

    # Tetris Parameters
    parser.add_argument("--data_path", type=str, help="Path to data file")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    data = pd.read_csv(args.data_path)

    print(data.shape)
    print(data.columns)

    is_training_data = (data["Replay_Memory_Full"] == True)

    training_data = data[is_training_data]

    print(training_data.shape)

    score_data = training_data["Tetrominoes"]
    loss_data = training_data["Average_Loss"]
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

    print(score_data_agg)

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

    print(seaborn_data.head())
    # sns.lineplot(data=score_data_agg, x="Game Number", y=["Tetromino Count", "Random Policy"])
    # sns.lineplot(x="Game Number", y="value", hue="variable", data=pd.melt(score_data_agg, ["Game Number"]))

    #sns.lineplot(data=seaborn_data, x="Game Number", y="Tetromino Count", hue="Policy")
    #sns.lineplot(data=cleared_line_data, x="Game Number", y="Cleared Lines", hue="Policy")
    #sns.lineplot(data=reward_data, x="Game Number", y="Reward Sum", hue="Policy")
    #sns.lineplot(data=loss_data, x="Game Number", y="Log Loss Value", hue="Average Log Loss")
    
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=seaborn_data, x="Game Number", y="Tetromino Count", hue="Policy", ax=ax1)

    fig2, ax2 = plt.subplots()
    sns.lineplot(data=loss_data, x="Game Number", y="Log Loss Value", hue="Average Log Loss", ax=ax2)
    
    fig3, ax3 = plt.subplots()
    sns.lineplot(data=cleared_line_data, x="Game Number", y="Cleared Lines", hue="Policy", ax=ax3)
    
    fig4, ax4 = plt.subplots()
    sns.lineplot(data=reward_data, x="Game Number", y="Reward Sum", hue="Policy", ax=ax4)
    
    fig5, ax5 = plt.subplots()
    sns.lineplot(data=action_count_data, x="Game Number", y="Action Count", hue="Policy", ax=ax5)

    plt.tight_layout()
    plt.show()

    # print("Cols = " + str(data.head()))