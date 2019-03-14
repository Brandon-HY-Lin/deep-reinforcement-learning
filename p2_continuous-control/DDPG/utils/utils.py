import numpy as np
import pickle
from collections import namedtuple

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
                     