#!/usr/bin/env python

#!/bin/python
from __future__ import print_function
import sys
import os
import argparse

class PBSJob(object):
    def __init__(self,args):
        # Check input actually exists
        if not os.path.isfile(args.infile):
            sys.exit('The input file does not exist!')
        # Make sure we have the correct extension
        infile = args.infile.split('.')
        self.software = None
        try:
            extension = infile[1]
        except IndexError:
            sys.exit('Wrong extension')
        if (extension == 'py'):
            print('Extension: ',extension,'; assuming a PySCF job')
            self.software = 'python'
        elif (extension == 'pmd') or (extension == 'pinp'):
            print('Extension: ',extension,'; assuming a promolden job')
            self.software = 'promolden-81-omp-uhf'
        else:
            sys.exit('Unknown extension make the appropriate changes and resubmit.')
     
        self.name      = infile[0]
        self.infile    = args.infile 
        self.threads   = args.nthreads
        self.partition = args.partition
        self.time      = args.time
        self.script    = self.name+'.sh'

    def write_pbs_script(self):
        with open(self.script,'w') as f:
           f.write("#!/bin/bash\n")
           f.write("#PBS -q batch \n")
           f.write("#PBS -N "+self.name+"\n")
           f.write("#PBS nodes=1:ppn="+str(self.threads)+"\n")
           f.write("#PBS -e "+self.name+".err\n")
           f.write("#PBS -o "+self.name+".log\n")
           f.write("#PBS -V \n")
           f.write("\n")
           if self.software == 'python':
               f.write("\n")

        print(self.software+' submission script written to '+self.script) 
        print('To submit, do "qsub '+self.script+'"') 
        #TODO: submit job

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Gaussian 16 or Q-Chem submit script on Grace.')
    parser.add_argument("infile", help="Gaussian 16 or Q-Chem input file")
    parser.add_argument("-nt", "--nthreads",metavar='N', help="CPUs-per-node",type=int,default=24)
    parser.add_argument("-p","--partition",metavar='P',help="partition",default="pi_hammes_schiffer")
    parser.add_argument("-t","--time",metavar='T',help="time (hours)",type=int,default=4)
    args = parser.parse_args()

    slurmjob = SlurmJob(args)
    slurmjob.write_slurm_script()

