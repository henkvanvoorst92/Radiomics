import pickle
import pandas as pd
import os,sys
import xgboost as xgb
from sklearn import model_selection
import sklearn
import time
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from Utils.Performance import *
from Utils.Utils import *
from Modelling.MachineLearning import MachineLearning

class ModelVarSelect(MachineLearning):
	def __init__(self, opt,verbal=False,addname='varselect'):
		super(ModelVarSelect, self).__init__()
		
		self.verbal = verbal
		self.addname = addname
		self.output_folder = opt.output_folder
		if self.output_folder is not None:
			if not os.path.exists(self.output_folder):
				os.makedirs(self.output_folder)

		self.init_options(opt)

		if self.output_folder is not None:
			store_opt_json(opt)
			
	def __call__(self,df_train,df_test,xvars,yvar,mdlname='XGB',hp=None,addname=None):
		#addname can be used to identify output files
		if addname is not None:
			self.addname = addname
		
		# defines self.x, self.y used to train and test
		# inherited from MachineLearing
		self.set_train_test(df_train,df_test,xvars,yvar)
		
		#optimization inherited from MachineLearing taking the following steps: 
		#1 gridsearch with stratified (for yvar) cv to find best hp
		#2 uses self.scoring (by default r2 or AUC) to find optimal hp
		#3 refits model with best hp on all the training data
		self.mdlname, self.hp = mdlname,hp
		self.optimize_single_model()
		#compute importance metric, utilizes internally the self.best_mdl
		# to compute feature importance (RF) or gain (XGB)
		if self.output_type=='ordinal':
			self.ord_importance = self.ordinal_importance_metric(self.best_model)
			self.importance = pd.DataFrame(self.ord_importance['mean_res'])
		else:
			self.importance = self.importance_metric(self.best_model)
		#filter importance metric base on min value or percentile
		self.df_filtered = self.filter_importance(df_train,self.importance)
		
		self.store_results()
		
		return self.df_filtered, self.importance

	def filter_importance(self,df,importance):

		if self.importance_pct:
			val = np.percentile(importance,self.min_importance)
		else:
			val = self.min_importance
		importance = importance[(importance.values>val)]
		self.top_n_vars = importance.index
		out = df[self.top_n_vars]
		return out
	
	def store_results(self):
		print('--------','Storing results of',self.mdlname,'in',self.output_folder,'--------')
		p = os.path.join(self.output_folder,self.mdlname+'_filter')
		if not os.path.exists(p):
			os.makedirs(p)
		self.gs_res.to_excel(os.path.join(p,'1.gridsearch_results'+self.addname+'.xlsx'))
		self.res_train.to_excel(os.path.join(p,'2.train_results'+self.addname+'.xlsx'))
		self.res_test.to_excel(os.path.join(p,'3.test_results'+self.addname+'.xlsx'))
		self.importance.to_excel(os.path.join(p,'4.importance_scores_'+self.addname+'.xlsx'))
		self.df_filtered.to_excel(os.path.join(p,'5.filtered_input_data'+self.addname+'.xlsx'))
		if self.mdlname=='XGB':
			self.best_model.save_model(os.path.join(p,self.mdlname+self.addname+'.json'))
		pickle.dump(self.best_model, open(os.path.join(p,self.mdlname+self.addname+'.pic') ,"wb"))

class CorrVarSelect(MachineLearning):
	def __init__(self, opt, verbal=False):
		super(CorrVarSelect, self).__init__()
		self.verbal = verbal
		if opt is not None:
			self.init_options(opt)


	def __call__(self, df, importance):    
		# correlation matrix
		self.cm = df.corr(method=self.corr_type)
		
		#filter colinear vars based on scores per var in score_dct
		if importance is not None:
			score_dct = importance.T.to_dict(orient='records')[0]
			self.tokeep, self.torm = self.corr_filter(self.cm.copy(), score_dct, self.max_corr)
			print('Variables to keep after correlation filtering:', len(self.tokeep))
			print('Variables to remove after correlation filtering:', len(self.torm))
			self.df_select = df[self.tokeep]
			### filter top_n radiomics if required
			self.df_select, self.top_n_vars = self.filter_importance(self.df_select,importance.loc[self.tokeep])
		
		if self.verbal:
			sns.heatmap(self.cm)
			plt.title('Correlation matrix '+self.corr_type)
			plt.show()

		if self.output_folder is not None:
			print('--------','saving results in:', self.output_folder,'--------')
			if not os.path.exists(self.output_folder):
				os.makedirs(self.output_folder)

			self.cm.to_excel(os.path.join(self.output_folder,'1.correlation_matrix.xlsx'))
			self.df_score.to_excel(os.path.join(self.output_folder,'2.high_corr_vars.xlsx'))
			self.df_select.to_excel(os.path.join(self.output_folder,'3.corr_selected_vars.xlsx'))

		return self.df_select

	def corr_filter(self, cm, score_dct, max_corr):

		diag_cm = self.diagonal_mat(cm)
		X,Y = np.where(diag_cm>max_corr)
		ixs, cols = cm.index, cm.columns

		# check the (fi) score of all colinear vars and report tokeep and toremove
		out = []
		for x,y in zip(X,Y):
			x1, x2 = ixs[x], cols[y]
			v1,v2 = score_dct[x1], score_dct[x2]
			if v1>v2:
				keep = x1
				remove = x2
			else:
				keep = x2
				remove = x1
			out.append([x1,x2,v1,v2,keep,remove])

		out = pd.DataFrame(out,columns=['x1','x2', 'v1', 'v2', 'keep', 'remove'])

		torm = out['remove']
		tokeep = list(out['keep'])
		tokeep.extend(list(torm[np.isin(torm,tokeep)]))

		# final filtering
		torm = [r for r in torm if r not in tokeep]
		tokeep = [k for k in cols if k not in torm]

		self.df_score = out
		return list(set(tokeep)),list(set(torm))

	def filter_importance(self, df,importance):

		top_n_vars = self.tokeep
		if self.filter_top_n_radiomics is not None:
			if self.split_shape_intensity_radiomics:
				self.top_n_shape, self.top_n_intensity, self.top_n_firstorder = \
				self.separate_radiomics_n_important_variables(importance, self.filter_top_n_radiomics)
				top_n_vars = [*self.top_n_shape, *self.top_n_intensity, *self.top_n_firstorder,
									*[c for c in top_n_vars if not 'original' in c]]
				print('Top n shape radiomics:', self.top_n_shape)
				print('Top n intensity radiomics:', self.top_n_intensity)
				print('Top n firstorder radiomics:', self.top_n_firstorder)
		elif self.filter_top_n is not None: # split all variables together
			top_n_vars = self.n_important_variables(importance,self.filter_top_n)
		
		self.filtered_importance = importance.loc[top_n_vars]
		self.filtered_importance = self.filtered_importance.sort_values(by=self.filtered_importance.columns[0],ascending=False)

		return df[top_n_vars], top_n_vars

	def n_important_variables(self, importance, top_n):
		return importance.iloc[:top_n].index

	def separate_radiomics_n_important_variables(self, importance, top_n):
		shape_vars, intensity_vars, firstorder_vars = split_radiomics_shape_intensity_firstorder(importance.index)
		
		imp_shape = importance.loc[shape_vars]
		imp_intensity = importance.loc[intensity_vars]
		imp_firstorder = importance.loc[firstorder_vars]

		top_shape = self.n_important_variables(imp_shape, top_n)
		top_intensity = self.n_important_variables(imp_intensity, top_n)
		top_firstorder = self.n_important_variables(imp_firstorder, top_n)
		return top_shape, top_intensity, top_firstorder

	def top_n_radiomics(self, importance, top_n):
		radiomic_vars = [v for v in df.columns if 'original_' in v]
		return importance.loc[radiomic_vars].iloc[top_n].index

	def diagonal_mat(self, cm):
		n,m = cm.shape
		m = np.tri(n,m,-1)==0
		cm[m] = np.NaN
		return cm
