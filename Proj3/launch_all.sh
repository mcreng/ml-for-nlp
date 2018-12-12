#!/bin/bash
job=$1
for epoch in 25 
do
   for dropout in 0.2 0.5 
   do
      for hidden in 128 256 
      do
         for embed in 128 256
         do
            for batch in 32
            do
               now=$(date +"%Y-%m-%d-%R:%S:%N")
               qsub -F "$job $epoch $dropout $hidden $embed $batch" launch.sh -N "$job"-"$epoch"-"$dropout"-"$hidden"-"$batch" -e error-"$job"-"$epoch"-"$dropout"-"$hidden"-"$embed"-"$batch"-"$now".log -o output-"$job"-"$epoch"-"$dropout"-"$hidden"-"$embed"-"$batch"-"$now".log -l walltime=12:00:00
            done
         done
      done
   done
done
