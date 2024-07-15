from sklearn.pipeline import Pipeline

class CustomPipeline(Pipeline):
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove any large attributes that are not necessary
        if 'memory' in state:
            state['memory'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)