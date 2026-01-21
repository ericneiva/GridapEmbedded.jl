#!/bin/bash
#PBS -P kr97
#PBS -q normal
#PBS -l walltime=00:30:00
#PBS -l ncpus=24
#PBS -l mem=96gb
#PBS -N distributed-agfem
#PBS -l software=Gridap.jl
#PBS -l wd

julia --project -O3 test/dev/mpi/runtests.jl
