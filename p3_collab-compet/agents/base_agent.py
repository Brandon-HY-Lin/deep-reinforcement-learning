
class BaseAgent():
    def __init__(self):
        pass
    

    def act(self, states):
        raise NotImplementedError('Not implement act()')

     
    def reset(self):
        raise NotImplementedError('Not implement reset()')
                                  
    def model_dicts(self):
        raise NotImplementedError('Not implement model_dicts()')