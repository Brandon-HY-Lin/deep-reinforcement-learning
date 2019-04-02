import numpy as np
import pickle
from collections import namedtuple
import matplotlib.pyplot as plt

import torch


ScoreParcels = namedtuple('ScoreParcels', ['comment', 'path_scores', 'color'])

def plot_scores(score_parcels, size_window=100, show_origin=False, alpha=1.0, baseline=0.5):
    
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
    plt.plot(np.arange(len(scores)), np.ones(len(scores)) * baseline, 'k--')
    
    plt.legend()
    plt.ylabel('Score')
    plt.xlabel('Episode #')            
    plt.show()
                     

def plot_scores_v2(score_parcels, size_window=100, max_len=None, 
                   show_origin=False, alpha=1.0, mode='valid', 
                   draw_vertical=False, show_episode_on_label=False,
                  baseline=0.5, margin=200):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
                        
    for comment, path_score, color in score_parcels:
        with open(path_score, 'rb') as f:
            scores = pickle.load(f)
    
            if max_len is None:
                max_len = len(scores)
                
            scores = scores[:max_len]
            
            moving_average = np.convolve(scores, np.ones((size_window,)) / size_window, mode=mode)
        
            x_baseline = None
            for index, s in enumerate(moving_average):
                if s >= baseline:
                    x_baseline = index
                    break
                    
            if show_episode_on_label is True and x_baseline is not None:
                comment = comment + ', passed at ep #{}'.format(x_baseline)
                        
            # draw vertical line that shows the first point is greater than 30.0
            if draw_vertical is True:
                len_vert = int(max(1, baseline))
                plt.plot(x_baseline * np.ones(len_vert), np.arange(len_vert), 
                         color=color, alpha=alpha*0.2)
                
                
                
              
            # draw moving average
            plt.plot(np.arange(len(moving_average)), moving_average,
                label=comment,
                color=color, alpha=alpha)
        
            if show_origin:
                plt.plot(np.arange(len(scores)), scores, 
                        color=color, alpha=alpha*0.25)
                
            
                
                
    # draw horizontal line
    plt.plot(np.arange(len(scores)), np.ones(len(scores)) * baseline, 'k--')
    
    
    
    ax.set_xlim(0, max_len+margin)
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
        
        
