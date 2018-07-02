#!/bin/bash
#PBS -N job
#PBS -q batch
#PBS -l nodes=1:ppn=12
#PBS -e job.err
#PBS -o job.log
#PBS -V
module load use.own
module load qc/1.0.0 
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=1
export OMP_STACKSIZE=2048m
export scratch_ssd="/scratch-ssd/jluis"
export TMPDIR=${scratch_ssd}
name="cr2"
cd ${PBS_O_WORKDIR}
python ${name}.py > ${name}.out
exit 0
