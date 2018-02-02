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


X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)

pdb.set_trace()

class MyModule(nn.Module):
	def __init__(self, num_units=10):
		super(MyModule, self).__init__()
		
		self.dense0 = nn.Linear(20, num_units)
		self.dropout = nn.Dropout(0.5)
		self.dense1 = nn.Linear(num_units, 10)
		self.output = nn.Linear(10, 2)
	def forward(self, X, **kwargs):
		X = F.relu(self.dense0(X))
		X = self.dropout(X)
		X = F.relu(self.dense1(X))
		X = F.softmax(self.output(X), dim=-1)
		return X

net = NeuralNetClassifier(
	MyModule,
	optimizer=torch.optim.Adam,
	max_epochs=10,
	lr=0.01,
	use_cuda=False)

# params = {
#     'lr': [0.01, 0.02],
#     'max_epochs': [10, 20],
#     'module__num_units': [10, 20],
# }

# gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')
# gs.fit(X, y)
# print(gs.best_score_, gs.best_params_)

time_start = time.time()
net.fit(X,y)
y_proba = net.predict_proba(X)

print(time.time() - time_start)
		

