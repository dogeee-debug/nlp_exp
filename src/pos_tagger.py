'''
Author: dogeee-debug gxyhome030404@gmail.com
Date: 2025-11-21 22:55:17
LastEditors: dogeee-debug gxyhome030404@gmail.com
LastEditTime: 2025-11-22 15:56:42
Description: 实现CRF词性标注
'''

import sklearn_crfsuite
import joblib

class CRFPOSTagger:
    def __init__(self):
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=200,
            all_possible_transitions=True
        )

    def _sent2features(self, sent):
        features = []
        for i, (token, _) in enumerate(sent):
            feats = {
                'bias': 1.0,
                'char': token,
                'is_first': i == 0,
                'is_last': i == len(sent) - 1,
                'is_upper': token.isupper(),
                'is_digit': token.isdigit(),
                'prev_char': '' if i == 0 else sent[i-1][0],
                'next_char': '' if i == len(sent)-1 else sent[i+1][0]
            }
            features.append(feats)
        return features

    def _sent2labels(self, sent):
        return [label for _, label in sent]

    def train(self, train_sents):
        X_train = [self._sent2features(s) for s in train_sents]
        y_train = [self._sent2labels(s) for s in train_sents]
        self.model.fit(X_train, y_train)

    def predict(self, sent):
        if len(sent) > 0 and isinstance(sent[0], str):
            sent = [(c,'') for c in sent]

        X = [self._sent2features(sent)]
        pred = self.model.predict(X)
        return pred[0]
    
    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)
