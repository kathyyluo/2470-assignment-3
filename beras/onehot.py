import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        self.classes = np.unique(data)
        vectors = np.eye(len(self.classes))

        self.mapping = {}
        for i, label in enumerate(self.classes):
            self.mapping[label] = vectors[i]

    def forward(self, data):
        if not hasattr(self, "mapping"):
            self.fit(data)

        encoded = []
        for label in data:
            encoded.append(self.mapping[label])

        return np.array(encoded)

    def inverse(self, data):
        indices = np.argmax(data, axis=-1)
        return self.classes[indices]
