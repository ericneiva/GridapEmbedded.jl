using Test
using Gridap
using Gridap.ReferenceFEs
using GridapEmbedded
using GridapEmbedded.Interfaces

φ(x) = √( x[1]^2.0 + x[2]^2.0 ) - 1.0
∇φ(x) = VectorValue( x[1] / √( x[1]^2.0 + x[2]^2.0 ) ,
                     x[2] / √( x[1]^2.0 + x[2]^2.0 ) )

phi = AlgoimCallLevelSetFunction(φ,∇φ)

order = 1

n = 12
domain = (-1.25,1.25,-1.25,1.25)

partition = (n,n)
h = (domain[2]-domain[1])/n
bgmodel = CartesianDiscreteModel(domain,partition)
Ωbg = Triangulation(bgmodel)

degree = order == 1 ? 3 : 2*order

vquad = Quadrature(algoim,phi,degree,phase=IN)
Ωᵃ,dΩᵃ = TriangulationAndMeasure(Ωbg,vquad)

x = dΩᵃ.quad.cell_point.values
x = lazy_map(Reindex(x),dΩᵃ.quad.cell_point.ptrs)
writevtk(x,"algoim_quad_points")

Ωˢ = BoundaryTriangulation(Ωᵃ)
writevtk(Ωˢ,"outer_surrogate")

d(x)   = φ(x) * ∇φ(x)