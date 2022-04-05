
import pandas as pd
import numpy as np
from sklearn import metrics

def binary_outcome_metrics(y_true,y_prob, thres=0.5):#score=y_pred
		y_pred=(y_prob>thres)*1

		tn, fp, fn, tp=metrics.confusion_matrix(y_true, y_pred).ravel()
		
		accuracy = round(((tp+tn)/(tn+fp+fn+tp))*100,2)
		recall = round(((tp)/(fn+tp))*100,2) # =TPR
		specificity = round(tn / (tn+fp)*100, 2) # =TNR
		precision = round(((tp)/(tp+fp))*100,2) # =PPV
		NPV = round(((tn)/(tn+fn))*100,2)

		#fpr, tpr, thresholds = metrics.roc_curve(y_true, score)
		roc_auc = round(metrics.roc_auc_score(y_true, y_pred)*100,2)
		roc_auc_prob = round(metrics.roc_auc_score(y_true, y_prob)*100,2)
		brier = metrics.brier_score_loss(y_true,y_prob)
		f1 = metrics.f1_score(y_true, y_pred)
		
		out = {'acc':accuracy, 
					 'TPR': recall, 'TNR':specificity, 
					 'PPV':precision, 'NPV': NPV,
					 'AUC':roc_auc, 
					 'AUC_prob':roc_auc_prob, 
					 'brier':brier,
					 'F1': f1, 
					 'y_pred=1_%':round(y_pred.sum()/y_pred.shape[0]*100,2), 
					 'y_true=1_%':round(y_true.sum()/y_true.shape[0]*100,2),
					 'y_pred=1_abs':y_pred.sum(),
					 'y_true=1_abs':y_true.sum()}
		
		return out #accuracy, recall, precision, specificity, roc_auc#, fpr, tpr, y_pred, y_prob

def regression_outcome_metrics(y_true,y_pred, thres=0.5):
	metr = {
					'explained_variance':metrics.explained_variance_score,
					'max_error':metrics.max_error,
					'neg_mean_absolute_error':metrics.mean_absolute_error,
					'neg_mean_squared_error':metrics.mean_squared_error,
					#'neg_root_mean_squared_error':metrics.mean_squared_error,
					'neg_mean_squared_log_error':metrics.mean_squared_log_error,
					'neg_median_absolute_error':metrics.median_absolute_error,
					'r2':metrics.r2_score,
					'neg_mean_poisson_deviance':metrics.mean_poisson_deviance,
					'neg_mean_gamma_deviance':metrics.mean_gamma_deviance,
					'neg_mean_absolute_percentage_error':metrics.mean_absolute_percentage_error,
					
					}
	out = {}
	for n,m in metr.items():
		score = m(y_true,y_pred)
		out[n] = score
	return out

def custom_roc_auc(y_true, y_prob):
	return metrics.roc_auc_score(y_true,y_prob)