import numpy as np
from sklearn.datasets import make_classification
import torch
from torch import nn
import torch.nn.functional as F
from skorch.net import NeuralNetClassifier
from torch import optim

import time
import pdb

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=True, random_state=42)

count_vectorizer = CountVectorizer()
x_train_counts = count_vectorizer.fit_transform(twenty_train.data)

X_train_tfidf = TfidfTransformer().fit_transform(x_train_counts)
print(X_train_tfidf.shape)

X_train_tfidf = X_train_tfidf.astype(np.float32).toarray() # Otherwise cant fit because its sparse

class NlpModel(nn.Module):
    def __init__(self, nb_features=35788, nb_out=len(categories), doprob=0.2):
        super(NlpModel, self).__init__()
        self.dense0 = nn.Linear(nb_features, 512)
        self.dropout = nn.Dropout(doprob)
        self.dense1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(doprob)
        self.dense2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(doprob)
        self.dense3 = nn.Linear(128, nb_out)

    def forward(self, X):
        X = F.relu(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = self.dropout(X)
        X = F.relu(self.dense2(X))
        X = self.dropout(X)
        X = F.softmax(X, dim=-1)

        return X


SAVE_MODEL = True
PRETRAINED_MODEL = True
RETRAIN_MODEL = False
SEARCH_FOR_PARAMS = False

net = NeuralNetClassifier(
    NlpModel,
    max_epochs=25,
    optimizer=torch.optim.Adam,
    lr=0.005,
    use_cuda=False)

if (PRETRAINED_MODEL):
    try:
        net.initialize()
        net.load_params('toy_model.pkl')
    except:
        print('No compatible precomputed model with name "toy_model.pkl" found. Training new model.')
        PRETRAINED_MODEL = False

if (not PRETRAINED_MODEL or (PRETRAINED_MODEL and RETRAIN_MODEL)):
    if(SEARCH_FOR_PARAMS):
        params = {
        'lr': [0.01, 0.005, 0.001],
        'max_epochs': [10, 20]}
        gs = GridSearchCV(net, params, refit=False, scoring='accuracy')
        gs.fit(X_train_tfidf, twenty_train.target)

    else:
        net.fit(X_train_tfidf,  twenty_train.target)

if ((not PRETRAINED_MODEL and SAVE_MODEL) or (PRETRAINED_MODEL and RETRAIN_MODEL)):
    net.save_params('toy_model.pkl')

new_docs = ['PyTorch is my favorite deep learning and gpu library', 'Cure for cancer and other such diseases would be useful, especially in the form of cheap medicine accessible to everyone with the sickness.', 'Spinoza has defined God as an entity found everywhere, in logic, in love, in materials, etc.']

to_predict = count_vectorizer.transform(new_docs)
to_predict = TfidfTransformer().fit_transform(to_predict)

predicted = net.predict(to_predict.astype(np.float32).toarray())

for num, i in enumerate(predicted):
    print(new_docs[num][0:10] + '....')
    print('Predicted {} \n ------------------'.format(categories[i]))

pdb.set_trace()