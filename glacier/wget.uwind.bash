#!/bin/bash


for year in {1989..2009}
do
	wget ftp://ftp.cdc.noaa.gov/Datasets/NARR/monolevel/uwnd.10m.$year.nc
#	wget ftp://ftp.cdc.noaa.gov/Datasets/NARR/monolevel/vwnd.10m.$year.nc
done
