'''
Created on May 9, 2021

@author: William
'''
import os

ROOT_DIR = "C:/Users/William/eclipse-workspace-cs747/Tetris-Deep-Q-Network/output/dqn_v6"


def count_lines(episodes_file):
    with open(episodes_file) as f:
        num_lines = sum(1 for line in f)
    
    return num_lines
    

if __name__ == '__main__':
    dir_list = os.listdir(ROOT_DIR)
    for dir_entry in dir_list:
        logs_dir = os.path.join(ROOT_DIR, dir_entry, "logs")
        
        run_files = os.listdir(logs_dir)
        found = False
        for run_file in run_files:
            if run_file.endswith("episodes.csv"):
                episodes_file = os.path.join(logs_dir, run_file)
                num_lines = count_lines(episodes_file)
                print(run_file + " -- " + str(num_lines))
                found = True
        
        if not found:
            print("No episodes file found for " + dir_entry)
                