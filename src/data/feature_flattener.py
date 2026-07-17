class FeatureFlattener:

    @staticmethod
    def flatten(X):

        samples = X.shape[0]

        return X.reshape(samples, -1)