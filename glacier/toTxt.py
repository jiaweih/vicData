#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
import os
from netCDF4 import date2num, num2date
sys.path.append('/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')

def get_data_ts(ifile,nlat,nlon,start=None,end=None,timname='time'):
    if os.path.exists(ifile):
        nc = Dataset(ifile,'r')
    else:
        sys.exit('No such file:{}'.format(ifile))
    for key in nc.variables.keys():
        if key not in nc.dimensions.keys():
            var = key
    time =  num2date(nc.variables[timname][:],
                     nc.variables[timname].units,
                     nc.variables[timname].calendar)
    df = pd.Series(nc.variables[var][:,nlat,nlon],index = time)
    lons = nc.variables['lon'][:]
    lats = nc.variables['lat'][:]
    nc.close()
    if start:
        df = df[start:]
    if end:
        df = df[:end]
    return df, format(lats[nlat],'.5f'), format(lons[nlon],'.5f')

nlats = 24
nlons = 30
ifile = '/Volumes/HDTemp/jiawei/pacific/UNSR/{0}_{1}_{2}.nc'
odir = '/Volumes/HDTemp/jiawei/pacific/txt/UNSR/{0}/{1}'
ofile = '/Volumes/HDTemp/jiawei/pacific/txt/UNSR/{0}/{1}/data_{2}_{3}'
models = ['BCSD+ANUSPLIN300+CanESM2','BCSD+ANUSPLIN300+ACCESS1-0','BCSD+ANUSPLIN300+CNRM-CM5','BCSD+ANUSPLIN300+MRI-CGCM3',
'BCSD+ANUSPLIN300+MIROC5','BCSD+ANUSPLIN300+MPI-ESM-LR','BCSD+ANUSPLIN300+CSIRO-Mk3-6-0','BCSD+ANUSPLIN300+inmcm4',
'BCSD+ANUSPLIN300+GFDL-ESM2G','BCSD+ANUSPLIN300+HadGEM2-CC','BCSD+ANUSPLIN300+HadGEM2-ES','BCSD+ANUSPLIN300+CCSM4']
scenarios = ['historical_rcp45','historical_rcp85']
for model in models:
    for scenario in scenarios:
        odir_t = odir.format(model,scenario)
        if os.path.exists(odir_t):
            print 'Existing path: {0}'.format(odir_t)
        else:
            os.makedirs(odir_t)
        for nlat in range(nlats):
            for nlon in range(nlons):
                df_pr,lat,lon = get_data_ts(ifile.format(model,scenario,'pr'),nlat,nlon)
                df_tasmax,lat,lon = get_data_ts(ifile.format(model,scenario,'tasmax'),nlat,nlon)
                df_tasmin,lat,lon = get_data_ts(ifile.format(model,scenario,'tasmin'),nlat,nlon)
                array = np.array([df_pr.values,df_tasmax.values,df_tasmin.values])
                np.savetxt(ofile.format(model,scenario,lat,lon),array.transpose(),fmt='%3f')
