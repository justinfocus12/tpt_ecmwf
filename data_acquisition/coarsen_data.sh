# Remap raw input files to the desired resolution of 2.5 x 2.5 
# args: infile, outfile
module load cdo/intel/1.9.10
cdo remapbil,r73x144 $1 $2
