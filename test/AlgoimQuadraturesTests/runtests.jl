module AlgoimQuadraturesTests

using Test

@testset "AlgoimInterface" begin include("AlgoimInterfaceTests.jl") end

@testset "PoissonAlgoim" begin include("PoissonAlgoimTests.jl") end

# @testset "NarrowBandStokes" begin include("NarrowBandStokesTests.jl") end

end # module