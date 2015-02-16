#/bin/bash
# extract the area we want

odir_UNSR=/Volumes/HDTemp/jiawei/pacific/UNSR
odir_PLACE=/Volumes/HDTemp/jiawei/pacific/PLACE

for file in ./data/*
do
	fn=`basename $file`
	ncks -d lat,51.5,53.5 -d lon,-118.5,-116.0 $file $odir_UNSR/$fn
	ncks -d lat,50.0,51.0 -d lon,-124.0,-121.0  $file $odir_PLACE/$fn
done
