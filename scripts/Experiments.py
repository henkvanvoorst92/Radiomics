from options.options import get_options
import os
from Utils.Utils import *
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from sklearn import model_selection
import random

def instantiate_experiments(root, outcomes):
	opt = get_options()
	for outcomename, outcometype in outcomes:
	    #set outcome type binary or cont
	    opt.outcome_type = outcometype
	    opt.yvar = outcomename

	    if outcometype=='binary' or outcometype=='ordinal':
	        opt.custom_loss = 'binary:logistic'
	        opt.best_score = 'highest'
	    else:
	        opt.custom_loss = 'regr:squarederror'
	        opt.best_score = 'lowest'

	    if outcomename=='n_attempts':
	        opt.custom_loss = "count:poisson"
	    #define higher/lower as better

	    #define output folder
	    opt.output_folder = os.path.join(root,outcomename)
	    if not os.path.exists(opt.output_folder):
	        os.makedirs(opt.output_folder)
	        
	    store_opt_json(opt)

def import_data(path, p_icc, icc_thresh=.75):
	df = pd.read_excel(path, engine='openpyxl')
	### standard preprocessing for every model
	cv_bin = ['sex',#'med_apt', 
	          'prev_str','prev_dm','ivtrom','prev_af',
	            *[c for c in df if 'occlsegment_c_short' in c]]
	cv_ord  = [ 'premrs','collaterals','ASPECTS_BL',]
	cv_cont = ['age1','NIHSS_BL',# 'INR', 
	           'rr_syst', 
	           'togroin', 'months'] #,'durproc',
	mtm =['thrombus_length', 'perviousness','mean_HU_over_markers']
	cont = [*cv_cont, *mtm]
	outcomes = ['total_attempts','FPR', 'ThreePR', 'GoodmRS', 
	'mrs_rev', 'attempts_to_succes', 'posttici_c', 'TICI_2B-3', 'n_attempt_0-3']
	clinical_xvars = [*cv_bin,*cv_ord,*cv_cont]

	radiomic_vars = [c for c in df if 'original_' in c]
	print('Total available radiomic variables:',len(radiomic_vars))
	#ICC = pd.read_excel(r'E:\Radiomics\ICC_score\icc2k.xlsx')
	ICC = pd.read_excel(p_icc)

	radiomic_vars = list(ICC[ICC.ICC>=icc_thresh].variable.values)
	print('Radiomic variables with ICC2k>0.75:',len(radiomic_vars))

	vars_not_norm = [*cv_bin, *cv_ord, *outcomes]
	vars_to_norm = [*cont, *radiomic_vars]
	xvars = [*cv_bin, *cv_ord, *cv_cont,*mtm, *radiomic_vars]
	print('Total x variables available:',len(xvars))
	print('Number of binary, ordinal, continuous, and manual thrombus measurement variables',
	      len(cv_bin), len(cv_ord),len(cv_cont), len(mtm))
	return xvars, [cv_bin, cv_ord, cv_cont,mtm, radiomic_vars, vars_to_norm], outcomes, df

def train_test_ID(IDs, test_size=200,seed=None,root_store=None):
	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)
	IDs = IDs.copy()
	np.random.shuffle(IDs)

	testID = IDs[:test_size]
	trainID = IDs[test_size:]
	if root_store is not None:
		if not os.path.exists(root_store):
			os.makedirs(root_store)
		train_pid = os.path.join(root_store,'train_ID.xlsx')
		trainID.to_excel(train_pid)

		test_pid = os.path.join(root_store,'test_ID.xlsx')
		testID.to_excel(test_pid)

	return testID, trainID

def split_train_test_norm(df,xvars,outcomes, vars_to_norm, test_size=100, scaler=RobustScaler(), seed=None):
	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)
	df = df.copy()
	df.index = df.ID
	#x,y = df[[*xvars,'ID']],df[[*outcomes,'ID']]
	x,y = df[xvars],df[outcomes]
	x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,
	                                                    stratify=y['mrs_rev'], 
	                                                    test_size=test_size)

	if scaler is not None:
		scaler.fit(x_train[vars_to_norm])
		x_train[vars_to_norm] = scaler.transform(x_train[vars_to_norm])
		x_test[vars_to_norm] = scaler.transform(x_test[vars_to_norm])
	
	df_train = x_train[xvars].merge(y_train, how='left', 
	                                         left_index=True,
	                                         right_index=True, 
									         ).T.drop_duplicates().T
	df_test = x_test[xvars].merge(y_test, how='left', 
	                                         left_index=True,
	                                         right_index=True, 
									         ).T.drop_duplicates().T
	#df_train = pd.concat([x_train[xvars],y_train],axis=1).T.drop_duplicates().T
	#df_test = pd.concat([x_test[xvars],y_test],axis=1).T.drop_duplicates().T
	print(x.shape,x_train.shape, x_test.shape, y_train.shape, y_test.shape)
	return df_train, df_test

# if train and test IDs are predefined
def load_train_test_norm(path_train_ID, path_test_ID, df,xvars,outcomes, vars_to_norm, scaler=RobustScaler()):
	df = df.copy()
	df.index = df.ID
	trainID = list(pd.read_excel(path_train_ID).ID.values.flatten())
	testID = list(pd.read_excel(path_test_ID).ID.values.flatten())

	x,y = df[xvars],df[outcomes]
	#x,y = df[xvars],df[outcomes]
	x_train, x_test, y_train, y_test = x.loc[trainID], x.loc[testID], y.loc[trainID], y.loc[testID]
	#x_train, x_test = [d.reset_index(drop=True)for d in [x_train, x_test]]
	#y_train, y_test = y_train.reset_index(), y_test.reset_index()

	if scaler is not None:
	    scaler.fit(x_train[vars_to_norm])
	    x_train[vars_to_norm] = scaler.transform(x_train[vars_to_norm])
	    x_test[vars_to_norm] = scaler.transform(x_test[vars_to_norm])

	df_train = x_train[xvars].merge(y_train, how='left', 
                                         left_index=True,
                                         right_index=True,  
                                         ).T.drop_duplicates().T

	df_test = x_test[xvars].merge(y_test, how='left', 
                                         left_index=True,
                                         right_index=True, 
                                         ).T.drop_duplicates().T
	#pd.concat([x_train[xvars],y_train],axis=1).T.drop_duplicates().T
	#df_test = pd.concat([x_test[xvars],y_test],axis=1).T.drop_duplicates().T
	#df_train.index = df_train.ID
	#df_test.index = df_test.ID
	print(x.shape,x_train.shape, x_test.shape, y_train.shape, y_test.shape)
	return df_train, df_test

def run_experiment(MVS,CVS,ML,opt,
                   df_train,df_test, xvars, yvar,hp, 
                   mdlname='XGB',addname='', skip_filter=False,
                   finmdlname=None, finhp=None                  
                  ):

    if finmdlname is None:
        finmdlname = mdlname
    if finhp is None and finmdlname==mdlname:
        finhp = hp
    
    if not skip_filter:
        df_filter,importance = MVS(df_train,df_test,xvars,yvar,mdlname=mdlname,hp=hp,addname='_filter_search'+addname)
        #filter highly correlated features
        #idct = importance.T.to_dict(orient='records')[0] #used for filtering based on correlation
        df_filter = CVS(df_filter,importance)
        xvars_filtered = list(set(df_filter.columns))
        #add yvar back to filtered
        df_filter = pd.concat([df_filter,df_train[yvar]],axis=1)
    else:
        df_filter = df_train
        xvars_filtered = xvars
    print('Using',len(xvars_filtered), 'variables')
    print('Final model:',finmdlname)
    print('Final hp:',finhp)
    #train final XGB model
    ML.set_train_test(df_filter,df_test,xvars_filtered,yvar)
    test_res = ML(algos = {finmdlname:finhp}, addname='_final'+addname)
    return MVS,CVS,ML


