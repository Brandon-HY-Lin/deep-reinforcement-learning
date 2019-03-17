
class BaseAgent():
    def __init__(self):
        pass
    
    '''
        model_dict()
        
        Return:
            return name-model pair dict  example: ({'critic': critic_model, 
                                                    'actor': actor_model})
    '''
    def model_dicts(self):
        raise NotImplementedError('Error: In BaseAgent(), model_dicts() is not implemented')