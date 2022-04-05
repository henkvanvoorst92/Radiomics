import json
import numpy as np
import argparse
import os
import pandas as pd
import SimpleITK as sitk
from sklearn import preprocessing
from scipy import stats

def init_model_params(clf,params):
    for k,v in params.items():
        setattr(clf,k,v)
    return clf

def store_opt_json(opt):
    if opt.output_folder is not None:
        p_json = os.path.join(opt.output_folder,'opt.json')
        dct = vars(opt)
        for k,v in dct.items():
            if isinstance(v,type):
                dct[k] = str(v)
            elif isinstance(v,np.ndarray):
                dct[k] = str(v)
        with open(p_json, 'w', encoding='utf-8') as f:
            json.dump(dct, f, ensure_ascii=False, indent=4)

def load_opt_json(root, org_opt=None):
    p_json = os.path.join(root,'opt.json')
    with open(p_json) as f:
        dct = json.load(f)
    for k,v in dct.items():
        if k=='norm':
            dct[k] = getattr(nn,v.strip("<>''").split('.')[-1])
    dct['output_folder'] = root
    if org_opt is not None:
        for argname in vars(org_opt):
            if argname not in dct.keys():
                dct[argname] = getattr(org_opt,argname)
    return argparse.Namespace(**dct)

def one_vs_all_dumm(df, dropfirst=False, prefix='', nan_dummy_incl=True):
    df_out = pd.DataFrame(index=df.index)
    for c in df.columns:
        #if (datatype=='categorical')|(datatype=='nominal')|(datatype=='ordinal')|(datatype=='binary'):
        coldat = pd.get_dummies(df[c],prefix = c, 
                                prefix_sep= prefix,
                                dummy_na = nan_dummy_incl, 
                                drop_first = dropfirst)
        df_out = pd.concat([df_out, coldat], axis=1)
    return df_out

def np2itk(arr,original_img):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(original_img.GetSpacing())
    img.SetOrigin(original_img.GetOrigin())
    img.SetDirection(original_img.GetDirection())
    # this does not allow cropping (such as removing thorax, neck)
    #img.CopyInformation(original_img) 
    return img

def split_radiomics_shape_intensity_firstorder(variables):
    shape_vars = [v for v in variables if 'original_shape' in v]
    texture_vars = [v for v in variables if 'original_' in v and 'shape' not in v and 'firstorder' not in v]
    firstorder_vars = [v for v in variables if 'original_' in v and 'firstorder' in v]
    return shape_vars, texture_vars, firstorder_vars

def non_negative(df):
    minima = df.min()
    cols = minima[minima<0].index # colnames with negative minima
    df[cols] = df[cols]+abs(minima[minima<0])
    return df

# optionally scale the (continuous) parameters setting 5-95 percentiles as min-max
def RobScale(df, scaler = preprocessing.RobustScaler(quantile_range=(5,95))):
    scaler = scaler
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), columns = df.columns)
    return df

# scale with a transformation to z-scores
def Zscale(df, ddof=0):
    return stats.zscore(df, ddof=ddof, axis=0)



