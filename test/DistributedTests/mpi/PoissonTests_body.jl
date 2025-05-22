module MPITestsBody

using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include("../PoissonTests.jl")

if ! MPI.Initialized()
  MPI.Init()
end

function all_tests(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  t = PArrays.PTimer(ranks,verbose=true)
  PArrays.tic!(t)

  PoissonTests.main_cutfem(distribute,parts,cells=(16,16))
  PArrays.toc!(t,"Poisson")

  display(t)
end

if MPI.Comm_size(MPI.COMM_WORLD) == 4
  with_mpi() do distribute
    all_tests(distribute,(2,2))
  end
elseif MPI.Comm_size(MPI.COMM_WORLD) == 1
  with_mpi() do distribute
    all_tests(distribute,(1,1))
  end
else
  MPI.Abort(MPI.COMM_WORLD,0)
end

end #module
