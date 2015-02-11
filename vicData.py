#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
from datetime import datetime
import argparse


def main():
    in_dir,out_dir_data,out_dir_dates_prec,out_dir_dates_runbase = command_lines()
    for filename in os.listdir(in_dir):
    	in_path = os.path.join(in_dir,filename)
        df = read_files(in_path)
        write_data = calculate_data(df)
        write_files_data(write_data,out_dir_data,filename)
        dates_prec_max,dates_run_base_max = calculate_dates(df,write_data)
        write_files_dates(dates_prec_max,out_dir_dates_prec,filename)
        write_files_dates(dates_run_base_max,out_dir_dates_runbase,filename)

def read_files(in_path,startYear=1915,endYear=2006,startIndex=274,endIndex=33511):
    """
    Read data into DataFrame

    Inputs
    ---------
    startYear: starting calendar year
    endYear: end calendar year
    """
    start = datetime(startYear,10,1)
    end = datetime(endYear,9,30)
    dates = pd.date_range(start=start,end=end)
    header = ['prec','runoff','baseflow','run_base']
    data = np.genfromtxt(in_path)
    start_pos = startIndex - 1
    end_pos = endIndex
    prec = data[start_pos:end_pos,3]
    runoff = data[start_pos:end_pos,12]
    baseflow = data[start_pos:end_pos,13]
    run_base = runoff + baseflow
    data_extract = np.transpose([prec,runoff,baseflow,run_base])
    df = pd.DataFrame(data_extract, index=dates, columns = header)
    return df

def water_year(in_date):
    """
    Given certain date, calculate corresponding water year
    """
    if in_date.year%4 == 0:
        if in_date.timetuple().tm_yday <= 274:
            return in_date.year 
        else:
            return in_date.year + 1
    else:
        if in_date.timetuple().tm_yday <= 273:
            return in_date.year 
        else:
            return in_date.year + 1 

def calculate_data(df,startYear=1916,endYear=2006):
    """
    Calculate maximum annual prec, maximum annual run_base, maximum mean precipitation

    Inputs
    --------
    startYear: starting water year
    endYear: end water year

    Outputs
    --------
    Arrays containing years, annual maximum prec, annual maximum run_base, annual mean prec
    """
    years = np.arange(startYear,endYear+1)
    prec_max = df['prec'].groupby(lambda x:water_year(x)).max().values
    prec_mean = df['prec'].groupby(lambda x:water_year(x)).mean().values
    run_base_max = df['run_base'].groupby(lambda x:water_year(x)).max().values
    write_data = np.transpose([years,prec_max,run_base_max,prec_mean])
    return write_data

def calculate_dates(df,write_data):
    prec_max = write_data[:,1]
    run_base_max = write_data[:,2]
    dates_prec_max = get_dates(df,prec_max,'prec')
    dates_run_base_max = get_dates(df,run_base_max,'run_base')
    return dates_prec_max,dates_run_base_max

def get_dates(df,data,var,startYear=1916):
    """
    Calculate dates for maximum annual events

    Inputs
    --------
    startYear: starting water year

    Output
    --------
    Arrays containing year, month, day, annual maximum data
    """
    dates_max = np.empty([len(data),4])
    for i in np.arange(len(data)):
        index = df[df[var] == data[i]].index  ##There are more than two dates matching maximum annual values in some years 
        for j in np.arange(len(index)):
            if water_year(datetime(index[j].year,index[j].month,index[j].day)) == i + startYear:
                dates_max[i,:] = np.array([index[j].year,index[j].month,index[j].day,data[i]])
    return dates_max

def write_files_data(write_data,out_dir,filename):
    """
    Write data

    """
    if not os.path.exists(out_dir):
    	os.makedirs(out_dir)
    out_path = os.path.join(out_dir,filename)
    np.savetxt(out_path,write_data,fmt=('%i','%8.3f','%8.3f','%8.3f'))

def write_files_dates(write_dates,out_dir,filename):
    """
    Write dates for annual maximum events
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir,filename)
    np.savetxt(out_path,write_dates,fmt=('%4i','%4i','%4i','%8.3f'))

def command_lines():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir",
                        type=str,
                        help="Input directory",
                        required=True)
    parser.add_argument("-o", "--output_dir_data",
                        type=str,
                        help="Output data directory",
                        required=True)
    parser.add_argument("-prec", "--output_dir_dates_prec",
                        type=str,
                        help="Output dates directory for maximum annual precipitation",
                        required=True)
    parser.add_argument("-run_base", "--output_dir_dates_run_base",
                        type=str,
                        help="Output dates directory for maximum annual run_base",
                        required=True)
    args = parser.parse_args()
    return args.input_dir,args.output_dir_data,args.output_dir_dates_prec,args.output_dir_dates_run_base

if __name__ == "__main__":
	main()
