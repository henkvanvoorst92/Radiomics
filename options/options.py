import argparse

def get_options():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	#output folder
	parser.add_argument('--output_folder', default=r'', type=str, help='folder to store results')

	#input data
	parser.add_argument('--output_type', default='binary', type=str, help='continuous or binary for yvar definition')
	parser.add_argument('--xvars', default=None, help='list of column names in input df used to define x variables')
	parser.add_argument('--yvar', default=None, help='column name in df with target variable (y)')
	parser.add_argument('--random_seed', type=int, default=11, help='Random seed to fix reproducibility')
	parser.add_argument('--ordinal_thresh',type=bool, default=3, 
		help='If output_type is ordinal, all models are optimized for binary classification > the ordinal_thresh')

	#Random forest (RF) optimization variables
	parser.add_argument('--n_trees',type=int,default=100,help='Number of decision trees fit in the forest for optimization (ideally >1000, only used for XGB and RF)')
	parser.add_argument('--n_splits',type=int, default=5, help='Number of cross validation splits used for hp search (ideally >10, only used for XGB and RF)')
	parser.add_argument('--simple_best',type=bool, default=True, help='True pics the best performing hp settings without taking in regard statistical sign, currently not used')
	
	#optional argument to add
	parser.add_argument('--scoring', default=None, help='Dictionary of name metric and a sklearn.metrics metric used to find optimal cv results, first metric is the main one used')
	parser.add_argument('--best_score', type=str, default='highest', help='highest or lowest to depict if higher scoring metric is better (AUC) or lower (RMSE)')

	parser.add_argument('--min_icc',type=float, default=0.75, help='minimum icc')
	#Feature importance filtering of output
	parser.add_argument('--min_importance', default=.001, help='if a float variables with a feature importance/gain smaller than this are excluded')
	parser.add_argument('--importance_pct', type=bool,default=False, help='if True min_fi is percentile value for selection')
	parser.add_argument('--filter_top_n',default=100, help='if a number, filter to n variables for final model')
	parser.add_argument('--filter_top_n_radiomics',default=3, help='if a number, filter to n variables for final model')
	parser.add_argument('--split_shape_intensity_radiomics',default=True, help='if True and filter_top_n selects top n shape and intensity radiomic vars')
	parser.add_argument('--optimize_prob_thresh', default=True, help='Optimizes probability threshold for classification on training data')

	#correlation filtering
	parser.add_argument('--max_corr',type=float, default=0.6, help='maximum correlation between two variables before flagged')
	parser.add_argument('--corr_type',type=str, default='spearman', help='pearson for linear relation (org data), spearman for monotonic (rank order)')

	#Machine learning options
	parser.add_argument('--custom_loss',default="binary:logistic", help='if XGB is used this is the objective loss function')
	parser.add_argument('--cache_size',type=int,default=1000, help='some algo can define RAM cache size used')

	parser.add_argument('--bootstrap',type=int,default=100, help='Number of bootstrap samples used to report performance (only for XGB and RF)')
	parser.add_argument('--n_jobs',type=int,default=4, help='Number of cpu cores to use')

	opt,__ = parser.parse_known_args()
	return opt
