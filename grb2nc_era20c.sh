# From https://confluence.ecmwf.int/display/LDAS/Converting+from+grib+to+netCDF+with+cdo
datadir="/scratch/jf4241/ecmwf_data/era20c_data/2021-11-03"
echo $datadir
for i in {1900..2007}
do
    echo "Converting year ${i}"
    ip1=`expr $i + 1`
    prefix="${datadir}/${i}-11-01_to_${ip1}-04-30"
    grib_filename="${prefix}.grb"
    nc_filename="${prefix}.nc"
    cdo -f nc copy $grib_filename $nc_filename
done


