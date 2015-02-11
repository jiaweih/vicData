#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import os
import sys
import argparse
sys.path.append('/Library/Python/2.7/site-packages')
import lmoments as lm


dist_dict = {
            'parameter':{'PE3':lm.pelpe3,'GUM':lm.pelgum,'GEV':lm.pelgev,'GNO':lm.pelgno},
            'quantile':{'PE3':lm.quape3,'GUM':lm.quagum,'GEV':lm.quagev,'GNO':lm.quagno},
            'nparams':{'PE3':3,'GUM':3,'GEV':3,'GNO':3},
            'method':{'PE3':'Pearson III','GUM':'Gumbel','GEV':'GEV','GNO':'Log-Normal'}
            }
methods = ['PE3','GUM','GEV','GNO']
probs = np.array([0.01,0.1,0.2,0.3,0.5,0.75,0.90,0.95,0.99,0.998])
return_periods = np.array([2,10,25,50,100])

def main():	
    in_dir = command_lines()
    out_dir_figures = 'figures'
    out_dir_estimates = 'estimates'

    if not os.path.exists(out_dir_figures):
    	os.makedirs(out_dir_figures)

    if not os.path.exists(out_dir_estimates):
        os.makedirs(out_dir_estimates)

    for fn in os.listdir(in_dir):
    	in_file = os.path.join(in_dir,fn)
    	df = read_files(in_file)
    	out_file_figures = os.path.join(out_dir_figures,fn + '.pdf')
    	plot_figures(df,methods,probs,out_file_figures)
        out_file_estimates = os.path.join(out_dir_estimates,fn)
    	write_estimates(in_file,out_file_estimates)

def read_files(in_file):
    """
    Gringorten plotting position is used to calculate empirical probabilities
    """
    data = np.genfromtxt(in_file)
    df_t = pd.DataFrame(data, columns=['year','prec_max','flow','prec_mean'])
    df = df_t.sort(['flow'], ascending=False)
    num = len(data)
    df['rank'] = np.arange(1, num + 1)
    df['prob'] = (df['rank'] - 0.44)/(num + 0.12)
    return df

def lm_quantiles(data,method,probs):
    """ 
    calculate quantiles of given probabilities for each given distribution; L-moment is used to estimate 
    distribution parameters
    """
    npara = dist_dict['nparams'][method]
    lms = lm.samlmu(data,npara)
    paras = dist_dict['parameter'][method](lms)
    quantiles = dist_dict['quantile'][method](probs,paras)
    return quantiles

def estimate_floods(in_file,methods,return_periods):
    """
    estimate peak floods for given return_periods
    """
    data = np.genfromtxt(in_file)
    probs = 1 - 1.0/return_periods
    df = pd.DataFrame(index = return_periods)
    for method in methods:
    	npara = dist_dict['nparams'][method]
    	lms = lm.samlmu(data[:,2],npara)
    	paras = dist_dict['parameter'][method](lms)       
    	quantiles = dist_dict['quantile'][method](probs,paras)
    	df[method] = np.int_(quantiles)
    return df

def plot_figures(df,methods,probs,store_path):
    """
    plot exceedance probabilities for annual peak flow, distributions are fitted to flow data
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.semilogy(df['flow'],df['prob'],'ok',label='data')
    for method in methods:
        quantiles = lm_quantiles(df['flow'],method,probs)  
        label = dist_dict['method'][method]
        ax.semilogy(quantiles,1-probs,label=label)
    ax.set_xlabel('Annual Peak Flow')
    ax.set_ylabel('Probability of Exceedance')
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0.001)
    plt.savefig(store_path,dpi=300)

def write_estimates(in_file,out_file):
    df_estimates = estimate_floods(in_file,methods,return_periods)
    estimates = df_estimates.values
    index = np.reshape(df_estimates.index.values,(5,1))
    np.savetxt(out_file,('Years','PE3','GUM','GEV','GNO'),newline=' ',fmt='%5s')
    with open(out_file,'a') as f_handle:
        f_handle.write('\n')
        np.savetxt(f_handle,np.hstack((index,estimates)),fmt='%5i')

def command_lines():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir",
                        type=str,
                        help="Input directory",
                        required=True)
    args = parser.parse_args()
    return args.input_dir

if __name__ == "__main__":
    main()
