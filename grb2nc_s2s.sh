# From https://confluence.ecmwf.int/display/LDAS/Converting+from+grib+to+netCDF+with+cdo
module load cdo/intel/1.9.10
datadir="/scratch/jf4241/ecmwf_data/s2s_data/2021-11-01"
echo $datadir
for grib_filename in $datadir/hc*.grb
do
    nc_filename="${grib_filename/grb/nc}"
    cdo -f nc copy $grib_filename $nc_filename
    echo "converted to new filename ${nc_filename}"
done



