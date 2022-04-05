import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from Utils.Performance import *
import xgboost as xgb
from sklearn import model_selection
import sklearn
from sklearn import metrics
import time
from datetime import datetime
from Utils.Utils import *
from Modelling.Ordinal import OrdinalClassifier, ordinal_outcome_metrics

class MachineLearning(object):
	"""
	This class can be used to train and evaluate machine learning models
	How it works:
		1. instantiate the class, check option file and init option to see what should be defined
		2. call set_train_test to define the training and testing data
		3. call the class by passing a dictionary (algos) with:
			key:algoname (SVM,RF,XGB,LR) and
			value: the hyper parameter grid for the algo or None (uses default hp grid)
	"""

	def __init__(self, opt=None, verbal=False, addname=''):
		super(MachineLearning, self).__init__()
		#pm function to set all attr of option

		self.addname = addname
		self.verbal = verbal

		if opt is not None:
			self.init_options(opt)

	def init_options(self,opt):
		for arg in vars(opt):
		    setattr(self,arg, getattr(opt,arg))
		
		if self.output_folder is not None:
			if not os.path.exists(self.output_folder):
				os.makedirs(self.output_folder)
		#set default scoring if not defined
		if self.scoring is None: #sklearn.metrics scoring function for gridsearch and optimal model select
			if self. output_type=='binary' or self.output_type=='ordinal':
				self.scoring = {'AUC': 'roc_auc'}
				self.refit_arg = list(self.scoring.keys())[0]
				self.eval_metric = self.refit_arg
				#self.scoring = {'AUC_custom':}
				#self.scoring = {'F1': 'f1'}

				# self.scoring = metrics.make_scorer(custom_roc_auc, greater_is_better=True)
				# self.refit_arg = True
				# self.eval_metric = 'score'
				self.best_score = 'highest'
			elif self. output_type=='continuous':
				self.scoring = {'r2': 'r2'}
				self.best_score = 'lowest'
				self.refit_arg = list(self.scoring.keys())[0]


	#run set train test data before call to instantiate data to use per algo
	def set_train_test(self,df_train,df_test,xvars,yvar):
		self.xvars, self.yvar = xvars,yvar
		
		self.x,self.y = self.set_x_y(df_train)
		if df_test is None:
			df_test = df_train
		self.x_test,self.y_test = self.set_x_y(df_test)

	def __call__(self,algos = {'XGB':None}, addname=None):
		#addname can be used to identify output files
		if addname is not None:
			self.addname = addname

		if self.x is None or self.y is None:
			print('Error set_train_test should be called first')

		# algorithms to fit as key [SVM,XGB,etc] with hyperparameter grid as value, if none default hp grid used
		self.algos = algos 
		for self.mdlname,self.hp in self.algos.items():
			self.optimize_single_model()

			if self.output_type=='ordinal':
				self.importance = self.ordinal_importance_metric(self.best_model)
			else:
				self.importance = self.importance_metric(self.best_model)

			if self.output_folder is not None:
				self.store_results()

		return self.res_test

	def optimize_single_model(self):
		print('--------','starting analysis of', self.mdlname,self.addname,'--------')
		#define model gets right sklearn model and checks and adjusts hp internally
		mdl = self.define_model(self.mdlname)
		print('--------','gridsearch', self.mdlname,self.addname,'--------')
		t1 = time.time()
		#set default hp if hp is none
		self.hp_gridsearch(mdl,self.hp)
		self.get_best_hp()
		t2 = time.time()
		print('--------','gridsearch runtime',self.mdlname,self.addname, round(t2-t1),'--------')
		#train final model (if tree-based multiply n_trees *10)
		self.train_best_model(mdl,self.best.params.values[0])
		#compute performance
		self.res_train = self.results(self.best_model,self.x,self.y)
		self.res_test = self.results(self.best_model,self.x_test,self.y_test)
		# for mean/median std/IQR reporting purposes perform bootstrapping
		if self.bootstrap>0:
			self.res_train_bootstrap = self.results_bootstrap(self.best_model,self.x,self.y,self.bootstrap)
			self.res_test_bootstrap = self.results_bootstrap(self.best_model,self.x_test,self.y_test,self.bootstrap)

		t3 = time.time()
		print('--------','Final model and results',self.mdlname,self.addname,round(t3-t2),'--------')

	def store_results(self):
		print('--------','Storing results of',self.mdlname,'in',self.output_folder,'--------')
		p = os.path.join(self.output_folder,self.mdlname)
		if not os.path.exists(p):
			os.makedirs(p)
		self.gs_res.to_excel(os.path.join(p,'1.gridsearch_results'+self.addname+'.xlsx'))
		self.res_train.to_excel(os.path.join(p,'2.train_results'+self.addname+'.xlsx'))
		self.res_test.to_excel(os.path.join(p,'3.test_results'+self.addname+'.xlsx'))
		if self.bootstrap>0:
			self.res_train_bootstrap.to_excel(os.path.join(p,'4.train_results_bootstrap'+self.addname+'.xlsx'))
			self.res_test_bootstrap.to_excel(os.path.join(p,'5.test_results_bootstrap'+self.addname+'.xlsx'))
		
		self.importance.to_excel(os.path.join(p,'6.variable_importance'+self.addname+'.xlsx'))
		self.save_model(os.path.join(p,'model_'+self.mdlname+self.addname))


		metrics.plot_roc_curve(self.best_model, self.x_test, self.y_test)
		sns.lineplot(np.linspace(0,1,21), np.linspace(0,1,21))
		plt.savefig(os.path.join(p,'test_set_auc_roc.tiff'))
		plt.show()      

	def set_x_y(self,df):
		x = df[self.xvars]
		y = df[self.yvar]
		return x,y
	
	def define_model(self,modelname):  
		if modelname=='XGB':#extreme gradient boosting
			mdl = self.define_XGB()
		elif modelname=='RF':#random forest
			mdl = self.define_RF()
		elif modelname=='SVM':#support vector machines
			mdl = self.define_SVM()
		elif modelname=='LR':#logistic or linear regression
			mdl = self.define_LR()
		else:
			print('Choose modelname from XGB,RF,SVM, or LR, current modelname is not an option:',modelname)

		return mdl
	
	def hp_gridsearch(self,mdl,hp):
		#run gridsearch for hyperparameter
		#allways call set_x_y first
		
		#define cross validation method
		cv_method = model_selection.StratifiedShuffleSplit(
				n_splits=self.n_splits, random_state=self.random_seed)
		# define gridsearch approach
		gs_cv = model_selection.GridSearchCV(mdl, param_grid=hp,
				scoring=self.scoring, cv=cv_method,
				n_jobs=self.n_jobs, refit=self.refit_arg)
		
		#call gridsearch
		if self.output_type=='ordinal':
			y = (self.y.copy() > self.ordinal_thresh)
		else:
			y = self.y.copy()

		gs_cv.fit(self.x,y)
		self.gs_res = pd.DataFrame(gs_cv.cv_results_)
		self.gs_cv = gs_cv

	def get_best_hp(self,metric_eval=None):
		#define best metric to use for finding optimal settings
		if metric_eval is None:
			metric_eval = self.eval_metric

		# get top best settings (self.best) that fall within the range best-min_d (min_d minimum difference margin)
		if self.best_score=='highest':
			asc = False
		elif self.best_score=='lowest':
			asc = True
		else:
			print(self.best_score, 'is not a potential evaluation mode')
		self.gs_res = self.gs_res.sort_values(by='mean_test_'+metric_eval, ascending=asc)
		self.best = self.gs_res.head(1)
		self.best_row = self.best.head(1)
	
	def train_best_model(self,mdl,params):
		t1 = time.time()
		print('---------Start final model fit:',datetime.now(), '---------')
		
		params['n_estimators'] = self.n_trees
		params['n_jobs'] = self.n_jobs

		if self.mdlname=='XGB':
			if params['booster']=='gblinear':
				params['importance_type '] = 'weight'
			else:
				params['importance_type '] = 'gain'

		if self.mdlname=='RF':
			params['n_estimators'] = self.n_trees*10

		mdl = init_model_params(mdl,params)
		if self.output_type=='ordinal':
			mdl = OrdinalClassifier(mdl)

		mdl.fit(self.x,self.y) #self.y can be ordinal
		self.best_model = mdl        
		self.trainset_res = self.results(self.best_model,self.x,self.y) #this is train set res
		
		t2 = time.time()
		print('finished in:',round(t2-t1),'seconds')
	
	def optimal_probability_threshold(self):
		#optimal threshold based on training data!!
		
		if self.optimize_prob_thresh: # and s!='AUC'
			out = []
			thresholds = np.linspace(0,1,101)
			for t in thresholds:
				t = round(t,2)
				res = self.binary_res(self.best_model,self.x,self.y,t)
				out.append(res)

			out = pd.concat(out,axis=1)
			
			if self.eval_metric=='score':
				s = 'AUC'
			else:
				s = list(self.scoring.keys())[0]

			a = pd.DataFrame(thresholds).T
			a.columns = out.columns
			a.index = ['threshold']
			self.thresh_optimization = pd.concat([out,a],axis=0)

			
			ix = np.argmax(self.thresh_optimization.loc[s])

			self.optimal_thresh = self.thresh_optimization.iloc[:,ix].loc['threshold']
		else:
			self.optimal_thresh = .5

		return self.optimal_thresh


	def results(self,mdl,x,y):
		#compute results for classification or regression
		if self.output_type=='binary':
			thresh = self.optimal_probability_threshold()
			res = self.binary_res(mdl,x,y,thresh)

		elif self.output_type=='ordinal':
			res = self.ordinal_res(mdl,x,y)
		else:
			res = self.regression_res(mdl,x,y)
		return res

	def binary_res(self,mdl,x,y,thresh):
		if self.mdlname in ['RF','XGB']:
			y_pred = mdl.predict_proba(x)[:,1]
		else:
			y_pred = mdl.predict(x)
		res = binary_outcome_metrics(y,y_pred,thresh)
		res = pd.DataFrame(res.values(), index=res.keys(), columns=['score'])
		return res

	def ordinal_res(self,mdl,x,y):
		# threshold optimization not possible for ordinal
		y_pred = mdl.predict_proba(x)
		res = ordinal_outcome_metrics(y,y_pred)
		return res

	def regression_res(self,mdl,x,y):
		y_pred = mdl.predict(x)
		res = regression_outcome_metrics(y,y_pred)
		res = pd.DataFrame(res.values(), index=res.keys(), columns=['score'])
		return res

	def results_bootstrap(self,mdl,X,Y,n_resamples):
	    df = pd.concat([X,Y],axis=1)
	    #compute results for classification or regression
	    out_res = []
	    for i in range(n_resamples):
	        tmp = df.sample(n=100,replace=True)
	        x = tmp[self.xvars]
	        y = tmp[self.yvar]
	        if self.output_type=='binary':
	            if self.mdlname in ['RF','XGB']:
	                y_pred = mdl.predict_proba(x)[:,1]
	            else:
	                y_pred = mdl.predict(x)
	            res = binary_outcome_metrics(y,y_pred)

	        elif self.output_type=='ordinal':
	        	y_pred = mdl.predict_proba(x)
	        	res = ordinal_outcome_metrics(y,y_pred)

	        else:
	            y_pred = mdl.predict(x)
	            res = regression_outcome_metrics(y,y_pred)
	        out_res.append(list(res.values()))

	    if self.output_type=='ordinal':
	    	cols = res.index
	    else:
	    	cols = res.keys()
	    
	    out = pd.DataFrame(out_res,columns=cols)
	    return out
		
	def importance_metric(self, model):
		#importance metric per variable
		if self.mdlname == 'XGB':
			importance = self.get_gain(model)
		elif self.mdlname=='RF':
			importance = self.get_feature_importance(model)
		elif self.mdlname=='LR':
			importance = self.get_coef(model)
		elif self.mdlname=='SVM':
			importance = None
		else:
			print('Wrong model type implement or choose other:',self.mdlname)
			
		if self.verbal and self.mdlname!='SVM':
			sns.distplot(importance)
			plt.title(self.mdlname+' importance metric distribution')
			plt.show()
		return importance

	def ordinal_importance_metric(self,model):
		#computes ordinal performance metrics per variable, means over different thresholds
		out = []
		threshs = []
		for thresh,m in model.clfs.items():
		    out.append(self.importance_metric(m))
		    threshs.append(thresh)
		importance = pd.concat(out,axis=1)
		importance.columns = [col+'_'+str(num) for num,col in zip(threshs,importance.columns)]
		importance['mean_res'] = importance.mean(axis=1)
		importance = importance.sort_values(by='mean_res', ascending=False)
		return importance

	def get_feature_importance(self,model):
		# works only for sklearn RandomForest and Extratrees models
		importance = pd.DataFrame(data = model.feature_importances_, 
							index = self.xvars, columns=['fi'])
		importance = importance.sort_values(by='fi', ascending=False)
		return importance
	
	def get_gain(self,model):
		#works only for XGB
		importance = model.get_booster().get_score(importance_type='gain')
		importance = pd.DataFrame(data = importance.values(), index = importance.keys(), columns=['gain'])
		return importance.sort_values(by='gain', ascending=False)
	
	def get_coef(self,model):
		#works only for XGB
		importance = model.coef_[0]
		importance = pd.DataFrame(data = abs(importance), index = list(self.x.columns), columns=['coef'])
		return importance.sort_values(by='coef', ascending=False)
	
	def define_XGB(self):
		if self.output_type=='binary' or self.output_type=='ordinal':
			if self.custom_loss is None:
				mdl = xgb.XGBClassifier(random_state=self.random_seed,verbosity=0)
			else:
				mdl =  xgb.XGBClassifier(objective=self.custom_loss,random_state=self.random_seed,verbosity=0)
		elif self.output_type=='continuous':
			if self.custom_loss is None:
				mdl = xgb.XGBRegressor(random_state=self.random_seed,verbosity=0)
			else:
				mdl =  xgb.XGBRegressor(objective=self.custom_loss,random_state=self.random_seed,verbosity=0)
		else:
			print('Error unkown output type', self.output_type)
			
		if self.hp is None:
			self.hp = {
						'booster':['gbtree', 'gblinear', 'dart'],
						'learning_rate':[0.1,.3,.5], #regularization on step size (chang in trees)
						'gamma':[0,.5,1,2,10],#min loss reduction (similar to entropy)
						'max_depth':[3,6,9], #max tree depth
						'subsample':[.1,.3,.5,1],#sample % of instances
						'sampling_method':['uniform'],
						'lambda':[0,.1,1,2,5],
						'alpha':[0,.1,1,2,5],
						 }
			self.hp = {'booster':['gbtree'], #,'dart','gblinear'
				        'learning_rate':[.01,.1,.2,.3], #regularization on step size (chang in trees)
				        'gamma':[1,5,10,20,30], #[0,.5,1,2,10],#min loss reduction (similar to entropy)
				        'max_depth':[3,6,9], #max tree depth
				        'subsample':[.3,.5],#sample % of instances
				        'sampling_method':['uniform'],
				        #'tree_method':["gpu_hist"],
				        'lambda':[.1,.5,1,3], #[0,.1,1,2,5],
				        'alpha':[.1,.5,1,3], #[0,.1,1,2,5],
				         }

		return mdl
	
	def define_RF(self):
		if self.output_type=='binary' or self.output_type=='ordinal':
			mdl = sklearn.ensemble.ExtraTreesClassifier(random_state=self.random_seed)
			criteria = ['entropy']
		elif self.output_type=='continuous':
			mdl = sklearn.ensemble.ExtraTreesRegressor(random_state=self.random_seed)
			criteria = ['mse', 'mae']
		else:
			print('Error unkown output type', self.output_type)
			
		if self.hp is None:
			n_xvars = len(self.xvars)
			fracs = [int(n_xvars*f) for f in [.01,.05,.1,.15,.2,.3]]
			fracs.extend([int(np.sqrt(n_xvars)), int(np.log2(n_xvars))])
			fracs = list(set(fracs))
			if 0 in fracs:
				fracs.remove(0) # only unique non zero values
			#fracs = ['sqrt','log2',.01,.05,.1,.15,.20,.30]

			self.hp = {'min_impurity_decrease':[1e-4,7e-3,5e-3,3e-3,1e-3,1e-2],
					'criterion':criteria,
					'max_features':fracs,
					'min_samples_split':[25],
					'min_samples_leaf':[25],
					'class_weight':['balanced',None],
					'bootstrap':[True]}
		return mdl 
	
	def define_SVM(self):
		if self.output_type=='binary' or self.output_type=='ordinal':
			mdl =  sklearn.svm.SVC(random_state=self.random_seed, cache_size=self.cache_size)
			if self.hp is None:
				self.hp = {
						'C':[.1,.5,1,2,5],
						'kernel':['linear', 'poly', 'rbf', 'sigmoid']
						}        
		elif self.output_type=='continuous':
			mdl =  sklearn.svm.SVR(random_state=self.random_seed, cache_size=self.cache_size)
			if self.hp is None:
				self.hp = {
						'C':[.1,.5,1,2,5],
						'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
						'epsilon':[.01,.1,.3,.5,1]
						}
		else:
			print('Error unkown output type', self.output_type)

		return mdl
	
	def define_LR(self):
		
		if self.output_type=='binary' or self.output_type=='ordinal':
			mdl = sklearn.linear_model.LogisticRegression(random_state=self.random_seed)
			if self.hp is None:
				self.hp = {
							'penalty':['elasticnet'],
							'l1_ratio':[0,.1,.25,.5,.75,.9,1.0],
							'C':[.1,.5,1,2,5],
							'solver':['saga']
							}
				
		elif self.output_type=='continuous':
			mdl = sklearn.linear_model.ElasticNet(random_state=self.random_seed)
			if self.hp is None:
				self.hp = {
							'alpha':[0,0.1,.3,.5,1,2,5],
							'l1_ratio':[0,.1,.25,.5,.75,.9,1.0]
							}
		else:
			print('Error unkown output type', self.output_type)
		
		return mdl 

	def save_model(self,path):
		if self.mdlname=='XGB':
			self.best_model.save_model(path+'.json')
		pickle.dump(self.best_model, open(os.path.join(path+'.pic') ,"wb"))

	def load_model(self,path):
		# if self.mdlname=='XGB':
		# 	self.best_model.save_model(path)
		self.best_model = pickle.load(open(os.path.join(path) ,"rb"))


#best performance (feature select): {'alpha': 0.5, 'booster': 'dart', 'gamma': 0.2, 'lambda': 1, 
#'learning_rate': 0.15, 'max_depth': 9, 'sampling_method': 'uniform', 
#'subsample': 0.5, 'n_estimators': 1000, 'n_jobs': -1, 'importance_type ': 'gain'}
# best performance final ML model: {'alpha': 0.5, 'booster': 'gbtree', 'gamma': 0.3, 'lambda': 0, 
#'learning_rate': 0.1, 'max_depth': 3, 'sampling_method': 'uniform', 'subsample': 0.5, 
#'n_estimators': 1000, 'n_jobs': -1, 'importance_type ': 'gain'}
#14,530 seconds for this grid
# hp = {'booster':['gbtree','dart','gblinear'],
#         'learning_rate':[.001,.01,.05,0.1,.15], #regularization on step size (chang in trees)
#         'gamma':[0,.1,.2,.3,.4,.5], #[0,.5,1,2,10],#min loss reduction (similar to entropy)
#         'max_depth':[3,6,9], #max tree depth
#         'subsample':[.1,.3,.5,1],#sample % of instances
#         'sampling_method':['uniform'],
#         #'tree_method':["gpu_hist"],
#         'lambda':[0,.5,1,1.5], #[0,.1,1,2,5],
#         'alpha':[0,.5,1,1.5] #[0,.1,1,2,5],
#          } 

# #new fird:
# hp = {'booster':['gbtree'], #,'dart','gblinear'
#         'learning_rate':[.01,0.1,.15,.2], #regularization on step size (chang in trees)
#         'gamma':[1,2.5,5,10], #[0,.5,1,2,10],#min loss reduction (similar to entropy)
#         'max_depth':[3,6,9], #max tree depth
#         'subsample':[.3,.5],#sample % of instances
#         'sampling_method':['uniform'],
#         #'tree_method':["gpu_hist"],
#         'lambda':[.1,.5,1], #[0,.1,1,2,5],
#         'alpha':[.1,.5,1], #[0,.1,1,2,5],
#          } 
# #change this folder iteratively


