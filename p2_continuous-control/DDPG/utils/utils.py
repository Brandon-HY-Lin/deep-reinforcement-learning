import numpy as np
import pickle
from collections import namedtuple
import matplotlib.pyplot as plt

import torch


ScoreParcels = namedtuple('ScoreParcels', ['comment', 'path_scores', 'color'])

def plot_scores(score_parcels, size_window=100, show_origin=False, alpha=1.0):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
                        
    for comment, path_score, color in score_parcels:
        with open(path_score, 'rb') as f:
            scores = pickle.load(f)
    
            moving_average = np.convolve(scores, np.ones((size_window,)) / size_window, mode='valid')
            plt.plot(np.arange(len(moving_average)), moving_average,
                label=comment,
                color=color, alpha=alpha)
        
            if show_origin:
                plt.plot(np.arange(len(scores)), scores, 
                        color=color, alpha=alpha*0.5)
                
                
    # draw horizontal line
    plt.plot(np.arange(len(scores)), np.ones(len(scores)) * 30.0, 'k--')
    
    plt.legend()
    plt.ylabel('Score')
    plt.xlabel('Episode #')            
    plt.show()
                     

def log_path_name(dir_logs, version):
    return dir_logs + 'log_{}.pickle'.format(version)


def save_logs(scores, dir_logs, version):
    path_logs = log_path_name(dir_logs, version)
    
    with open(path_logs, 'wb') as f:
        pickle.dump(scores, f)
        

def save_agent(model_dicts, dir_checkpoints, version):
    for prefix_model_name, model in model_dicts.items():
        path_model = dir_checkpoints + 'checkpoint_{}_{}.pth'.format(prefix_model_name, version)
        
        torch.save(model.state_dict(), path_model)
        
        
