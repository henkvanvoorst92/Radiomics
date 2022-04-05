import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.base import clone
from Utils.Performance import binary_outcome_metrics

class OrdinalClassifier():
	
	def __init__(self, clf):
		self.clf = clf
		self.clfs = {}
	
	def fit(self, X, y):
		self.unique_class = np.sort(np.unique(y))
		if self.unique_class.shape[0] > 2:
			for i in range(self.unique_class.shape[0]-1):
				# for each k - 1 ordinal value we fit a binary classification problem
				binary_y = (y > self.unique_class[i]).astype(np.uint8)
				clf = clone(self.clf)
				clf.fit(X, binary_y)
				self.clfs[i] = clf
	
	def predict_proba(self, X):
		clfs_predict = {k:self.clfs[k].predict_proba(X) for k in self.clfs}
		predicted = []
		for i,y in enumerate(self.unique_class):
			if i == 0:
				# V1 = 1 - Pr(y > V1)
				predicted.append(1 - clfs_predict[y][:,1])
			elif y in clfs_predict:
				# Vi = Pr(y > Vi-1) - Pr(y > Vi)
				 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
			else:
				# Vk = Pr(y > Vk-1)
				predicted.append(clfs_predict[y-1][:,1])
		return np.vstack(predicted).T
	
	def predict(self, X): #just predicts max prob class
		return np.argmax(self.predict_proba(X), axis=1)

	def per_class_predict_proba(self, X): #just predicts max prob class
		#todo probably no improvement
		out = []
		for k,clf in self.clfs.items():
			y_pred = clf.predict_proba(X)[:,1]
			out.append(y_pred)
		return np.stack(out).T
	
def prob_per_threshold(y_pred):
    toty = np.zeros_like(y_pred)
    for i in range(y_pred.shape[1]):
        toty[:,i] = y_pred[:,i] + toty[:,i-1]
    return toty

def ordinal_outcome_metrics(y_true,y_pred, thresh=.5):
    ord_pred = np.argmax(y_pred, axis=1)

    unique_class = np.sort(np.unique(y_true))
    out = []
    for i in range(unique_class.shape[0]-1):
        # for each k - 1 ordinal value we fit a binary classification problem
        binary_y_true = (y_true > unique_class[i]).astype(np.uint8)
        binary_y_pred = (ord_pred > unique_class[i]).astype(np.uint8)
        res = binary_outcome_metrics(binary_y_true, binary_y_pred, thres=thresh)
        tmp = pd.DataFrame(res.values(), index=res.keys(), columns=['score_'+str(unique_class[i])])
        out.append(tmp)
    out = pd.concat(out,axis=1)
    out['mean_scores'] = out.mean(axis=1)
    return out



