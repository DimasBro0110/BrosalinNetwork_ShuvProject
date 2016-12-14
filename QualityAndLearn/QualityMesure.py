__author__ = 'DmitriyBrosalin'
from sklearn.cross_validation import train_test_split, cross_val_score


class Measurement:
    def __init__(self, data_X, data_Y, model):
        self.data_X = data_X
        self.data_Y = data_Y
        self.model = model

    def cross_validation_split(self):
        train_x, test_x, train_y, test_y = train_test_split(self.data_X, self.data_Y, test_size=0.25)
        return train_x, train_y, test_x, test_y

    def train_model(self):
        train_x, train_y, test_x, test_y = self.cross_validation_split()
        print("Start training model...")
        mlp = self.model.fit(train_x, train_y)
        scores = self.model.score(test_x, test_y)
        print("Model trained")
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores, 0.01))
        return mlp
