module DistributedAggregationMPITests

using Test
using MPI

mpidir = @__DIR__
testdir = joinpath(mpidir,"..")
repodir = joinpath(testdir,"..","..")

function run_test(procs,file)
  mpiexec() do cmd
    run(`nsys profile --trace=nvtx,mpi --mpi-impl=mpich $cmd -n $procs $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`)
  end
end

# Get the number of processes from command line arguments, default to 24
procs = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 24
run_test(procs,"runtests_body.jl")

end # module
