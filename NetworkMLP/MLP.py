from FeatureMaker.Features import Features
from QualityAndLearn.QualityMesure import Measurement

__author__ = 'DmitriyBrosalin'
from sknn.mlp import Classifier, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np


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
                  units=self.amount_of_features * 4, dropout=0.5),
            Layer(name="Hidden Layer 1", type="Sigmoid",
                  units=self.amount_of_features * 2, dropout=0.3),
            Layer(name="Hidden Layer 1", type="Sigmoid",
                  units=self.amount_of_features, dropout=0.3),
            Layer(name="Hidden Layer 1", type="Sigmoid",
                  units=self.amount_of_features / 4, dropout=0.3),
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

    def train_mlp(self, model):
        features_me = Features(
            "/Users/DmitriyBrosalin/Interesting/SpeechRecognition/Train_Me"
        ).create_features()
        features_stranger = Features(
            "/Users/DmitriyBrosalin/Interesting/SpeechRecognition/Train"
        ).create_features()
        stranger_label = np.array([0 for i in range(0, features_stranger.shape[0])])
        me_labels = np.array([1 for i in range(0, features_me.shape[0])])
        data_X = np.concatenate((features_stranger, features_me))
        data_y = np.concatenate((stranger_label, me_labels))
        quality = Measurement(data_X, data_y, model)
        quality = quality.train_model()
        return quality

    def predict_val(self, path_to_file, model):
        features = Features(path_to_file).create_vector_fetures(path_to_file)
        prediction = model.predict(features)
        return prediction