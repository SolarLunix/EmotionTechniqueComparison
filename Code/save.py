import pickle

class saveFunction():

    def save(self,name, features, labels):
        savedFiles = (features, labels)
        file = open('SavedFeatures/{0}.pkl'.format(name),'wb')
        pickle._dump(savedFiles, file)

    def load(self, name):
        pickle_in = open('SavedFeatures/{0}.pkl'.format(name), "rb")
        files = pickle.load(pickle_in)
        return files[0], files[1]