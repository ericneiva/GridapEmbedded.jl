#!/bin/bash
#PBS -P kr97
#PBS -q normal
#PBS -l walltime=00:30:00
#PBS -l ncpus=192
#PBS -l mem=768gb
#PBS -N distributed-agfem
#PBS -o /home/552/em5855/GridapEmbedded.jl/distributed-agfem-192.out
#PBS -e /home/552/em5855/GridapEmbedded.jl/distributed-agfem-192.err
#PBS -l software=Gridap.jl
#PBS -l wd

julia --project -O3 test/dev/mpi/runtests.jl 192
