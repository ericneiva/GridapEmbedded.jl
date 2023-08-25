module AlgoimTests

using Test

@testset "AlgoimInterface" begin include("AlgoimInterfaceTests.jl") end

@testset "PoissonAlgoim" begin include("PoissonAlgoimTests.jl") end

@testset "ClosestPoint" begin include("ClosestPointTests.jl") end

@testset "VolumeConservation" begin include("VolumeConservationTests.jl") end

@testset "Visualization" begin include("VisualizationTests.jl") end

end # module