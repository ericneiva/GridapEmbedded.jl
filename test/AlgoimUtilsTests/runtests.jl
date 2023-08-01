module AlgoimTests

using Test

@testset "AlgoimInterface" begin include("AlgoimInterfaceTests.jl") end

@testset "PoissonAlgoim" begin include("PoissonAlgoimTests.jl") end

@testset "ClosestPoint" begin include("ClosestPointTests.jl") end

end # module