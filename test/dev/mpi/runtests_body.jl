module DistributedAggregationMPI

using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include("../distributed_aggregation.jl")

const DA = DistributedAggregation

if ! MPI.Initialized()
  MPI.Init()
end

# problem = DA.disk
#
# if MPI.Comm_size(MPI.COMM_WORLD) == 25
#   with_mpi() do distribute
#     DA.run_benchmark_test(distribute,(5,5),10,problem)
#     for ncells_x_dir in (320,640,1280)
#       DA.run_benchmark_test(distribute,
#                             (5,5),
#                             ncells_x_dir,
#                             problem)
#     end
#   end
# elseif MPI.Comm_size(MPI.COMM_WORLD) == 100
#   with_mpi() do distribute
#     DA.run_benchmark_test(distribute,(10,10),20,problem)
#     for ncells_x_dir in (640,1280,2560)
#       DA.run_benchmark_test(distribute,
#                             (10,10),
#                             ncells_x_dir,
#                             problem)
#     end
#   end
# elseif MPI.Comm_size(MPI.COMM_WORLD) == 400
#   with_mpi() do distribute
#     DA.run_benchmark_test(distribute,(20,20),40,problem)
#     for ncells_x_dir in (1280,2560,5120)
#       DA.run_benchmark_test(distribute,
#                             (20,20),
#                             ncells_x_dir,
#                             problem)
#     end
#   end
# elseif MPI.Comm_size(MPI.COMM_WORLD) == 1600
#   with_mpi() do distribute
#     DA.run_benchmark_test(distribute,(40,40),80,problem)
#     for ncells_x_dir in (2560,5120,10240)
#       DA.run_benchmark_test(distribute,
#                             (40,40),
#                             ncells_x_dir,
#                             problem)
#     end
#   end
# elseif MPI.Comm_size(MPI.COMM_WORLD) == 6400
#   with_mpi() do distribute
#     DA.run_benchmark_test(distribute,(80,80),160,problem)
#     for ncells_x_dir in (5120,10240,20480)
#       DA.run_benchmark_test(distribute,
#                             (80,80),
#                             ncells_x_dir,
#                             problem)
#     end
#   end
# elseif MPI.Comm_size(MPI.COMM_WORLD) == 25600
#   with_mpi() do distribute
#     DA.run_benchmark_test(distribute,(160,160),320,problem)
#     for ncells_x_dir in (10240,20480,40960)
#       DA.run_benchmark_test(distribute,
#                             (160,160),
#                             ncells_x_dir,
#                             problem)
#     end
#   end
# end

problem = DA.popcorn

if MPI.Comm_size(MPI.COMM_WORLD) == 125
  with_mpi() do distribute
    DA.run_benchmark_test(distribute,(5,5,5),10,problem)
    for ncells_x_dir in (100,150,200)
      DA.run_benchmark_test(distribute,
                            (5,5,5),
                            ncells_x_dir,
                            problem)
    end
  end
elseif MPI.Comm_size(MPI.COMM_WORLD) == 1000
  with_mpi() do distribute
    DA.run_benchmark_test(distribute,(10,10,10),20,problem)
    for ncells_x_dir in (200,300,400)
      DA.run_benchmark_test(distribute,
                            (10,10,10),
                            ncells_x_dir,
                            problem)
    end
  end
elseif MPI.Comm_size(MPI.COMM_WORLD) == 8000
  with_mpi() do distribute
    DA.run_benchmark_test(distribute,(20,20,20),40,problem)
    for ncells_x_dir in (400,600,800)
      DA.run_benchmark_test(distribute,
                            (20,20,20),
                            ncells_x_dir,
                            problem)
    end
  end
end

end # module