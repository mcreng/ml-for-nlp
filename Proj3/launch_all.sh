#!/bin/bash
job=cnn
for epoch in 1 2 3 4 
do
   for dropout in `seq 0.7 0.1 0.9`
   do
      for hidden in 500
      do
         for embed in 100
         do
            qsub -F "$job $epoch $dropout $hidden $embed" launch.sh -N "$job"-"$epoch"-"$dropout"-"$hidden" -e error-"$job"-"$epoch"-"$dropout"-"$hidden"-"$embed".log -o output-"$job"-"$epoch"-"$dropout"-"$hidden"-"$embed".log -l walltime=12:00:00
         done
      done
   done
done
