
import six
import pandas as pd
from tqdm import tqdm
from radiomics import featureextractor
import os

def Radiomics2dict(result, include_diagnostics=True):
	dct = {}
	for key, val in six.iteritems(result):
		if 'diagnostics' in key and not include_diagnostics:
			continue
		dct[key] = val    
	return dct

def Radiomics2pandas(result, include_diagnostics=True):  
	ks, vals = [], []
	for key, val in six.iteritems(result):
		if 'diagnostics' in key and not include_diagnostics:
			continue
		ks.append(key)
		vals.append(val)
	return pd.Series(data=vals, index=ks)

def path_per_observer(path, observers):
    dct = {}
    for f in os.listdir(path):
        for n in observers:
            if n in f:
                dct[n] = os.path.join(path,f)
    return dct


### make a function to get all seg files in a df (or dct)

class ExtractRadiomics(object):
	def __init__(self, output_folder=None, param_file=None, verbal=False,addname=''):
		super(ExtractRadiomics, self).__init__()
		self.output_folder = output_folder
		self.addname = addname
		#custom yaml param file for feature extraction (or default for None)
		self.param_file = param_file

	def __call__(self, dct):

		df_out = self.extract_radiomics(dct, self.param_file)

		if self.output_folder is not None:
			print('--------','saving results in:', self.output_folder,'--------')
			if not os.path.exists(self.output_folder):
				os.makedirs(self.output_folder)
			df_out.to_excel(os.path.join(self.output_folder,'radiomic_features'+self.addname+'.xlsx'))
		# filter out non-variable columns (for reporting important)
		cols = [c for c in df_out if 'diagnostics' not in c]
		df_out = df_out[cols]
		return df_out

	def extract_radiomics(self,dct, param_file=None):
		#dct[key=ID]:values=[p_scan,p_seg]
		extractor = featureextractor.RadiomicsFeatureExtractor(param_file)

		self.out = []
		self.errors = []
		for ID,(p_scan,p_seg) in tqdm(dct.items()):
			print('Start Running:',ID)
			try:
				result = extractor.execute(p_scan, p_seg)
				sers = Radiomics2pandas(result)
				sers['diagnostics_ID'] = ID
				sers['diagnostics_path_seg'] = p_seg
				sers['diagnostics_path_ncct'] = p_scan
				self.out.append(sers)
			except:
				print('Error:',ID)
				self.errors.append(ID)
		#format output to row=ID by column=variables dataframe
		self.df_out = pd.concat(self.out,axis=1)
		self.df_out = self.df_out.T
		self.df_out.index = self.df_out['diagnostics_ID']
		return self.df_out


