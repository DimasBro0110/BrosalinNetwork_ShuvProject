from FeatureMaker.Features import Features
from NetworkMLP.MLP import MLP
import numpy as np
from QualityAndLearn.QualityMesure import Measurement

__author__ = 'DmitriyBrosalin'


if __name__ == "__main__":
    mlp = MLP(13)
    features_stranger = Features(
        "/Users/DmitriyBrosalin/Interesting/SpeechRecognition/Train"
    ).create_features()
    stranger_label = np.array([0 for i in range(0, features_stranger.shape[0])])
    features_me = Features(
        "/Users/DmitriyBrosalin/Interesting/SpeechRecognition/Train_Me"
    ).create_features()
    me_labels = np.array([1 for i in range(0, features_me.shape[0])])
    data_X = np.concatenate((features_stranger, features_me))
    data_y = np.concatenate((stranger_label, me_labels))
    for i in data_X:
        print(i)
    model = mlp.build_mlp()
    quality = Measurement(data_X, data_y, model)
    quality.train_model()
