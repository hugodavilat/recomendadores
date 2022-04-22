import pickle
import os

def pickle_cached(path):
    def decorator(func):
        def inner(arg):
            if (os.path.isfile(path)):
                r = pickle.load(open(path, 'rb'))
                return r
            else:
                r = func(arg)
                pickle.dump(r, open(path, "wb" ), pickle.HIGHEST_PROTOCOL)
                return r
        return inner
    return decorator