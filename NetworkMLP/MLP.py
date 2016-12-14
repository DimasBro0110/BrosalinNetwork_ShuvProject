__author__ = 'DmitriyBrosalin'
from sknn.mlp import Classifier, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class MLP:
    def __init__(self, amount_of_features):
        self.amount_of_features = amount_of_features
        self.amount_of_layers = amount_of_features * 2 / 3


    def __build_layers(self):
        print("Building layers...")
        layers = [
            Layer(name="Input Layer", type="Sigmoid",
                  units=self.amount_of_features),
            Layer(name="Hidden Layer 1", type="Sigmoid",
                  units=self.amount_of_features * 3, dropout=0.6),
            Layer(name="Hidden Layer 2", type="Sigmoid",
                  units=self.amount_of_features * 2, dropout=0.6),
            Layer(name="Hidden Layer 3", type="Sigmoid",
                  units=self.amount_of_features, dropout=0.4),
            Layer(name="Hidden Layer 4", type="Sigmoid",
                  units=self.amount_of_features / 2, dropout=0.4),
            Layer(name="Hidden Layer 5", type="Sigmoid",
                  units=self.amount_of_features / 4, dropout=0.25),
            Layer(name="Output", type="Softmax", units=2)
        ]
        print("Layers built successfully")
        return layers


    def build_mlp(self):
        layers = self.__build_layers()
        print("Building neural network...")
        pipeline = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network', Classifier(layers=layers, learning_rule='nesterov'))])
        print("Network built successfully")
        return pipeline