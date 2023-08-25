module VolumeConservationTests

using Test
using CxxWrap
using Gridap
using GridapEmbedded

using Gridap.ReferenceFEs

const IN = -1
const OUT = 1
const CUT = 0

order = 1
domain = (-1.1,1.1,-1.1,1.1)

f(x) = (x[1]*x[1]/(0.5*0.5)+x[2]*x[2]/(0.5*0.5)) - 1.0
function gradf(x::V) where {V}
    V([2.0*x[1]/(0.5*0.5),2.0*x[2]/(0.5*0.5)])
end

function run_case(n::Int,degree::Int,cppdegree::Int)

  partition = Int32[n,n]
  bgmodel = CartesianDiscreteModel(domain,partition)
  Ω = Triangulation(bgmodel)

  reffeᵠ = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(Ω,reffeᵠ)

  fₕ = interpolate_everywhere(f,V)
  phi = AlgoimCallLevelSetFunction(fₕ,∇(fₕ))

  squad = Quadrature(algoim,phi,degree,phase=CUT)
  dΓ = Measure(Ω,squad,data_domain_style=PhysicalDomain())
  xΓ = dΓ.quad.cell_point.values
  xΓ = lazy_map(Reindex(xΓ),dΓ.quad.cell_point.ptrs)
  # writevtk(xΓ,"sres_1")

  vquad = Quadrature(algoim,phi,degree,phase=IN)
  dΩ = Measure(Ω,vquad,data_domain_style=PhysicalDomain())
  vol_1 = ∑(∫(1)dΩ)

  cps = compute_closest_point_projections(Ω,phi,cppdegree=cppdegree)

  reffeᶜ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  W = TestFESpace(Ω,reffeᶜ)

  g(x) = VectorValue(1.0,0.0)
  gₕ = interpolate_everywhere(g,W)

  dt = 1.0
  _phi₂ = compute_normal_displacement(cps,phi,gₕ,dt,Ω)
  _phi₂ = get_free_dof_values(fₕ) - _phi₂
  _phi₂ = FEFunction(V,_phi₂)
  # writevtk(Ω,"bres_2",cellfields=["phi"=>_phi₂])
  phi₂ = AlgoimCallLevelSetFunction(_phi₂,∇(_phi₂))

  squad = Quadrature(algoim,phi₂,degree,phase=CUT)
  dΓ₂ = Measure(Ω,squad,data_domain_style=PhysicalDomain())
  xΓ = dΓ₂.quad.cell_point.values
  xΓ = lazy_map(Reindex(xΓ),dΓ₂.quad.cell_point.ptrs)
  # writevtk(xΓ,"sres_2")

  vquad = Quadrature(algoim,phi₂,degree,phase=IN)
  dΩ₂ = Measure(Ω,vquad,data_domain_style=PhysicalDomain())
  vol_2 = ∑(∫(1)dΩ₂)

  error = abs(vol_1-vol_2)/abs(vol_1)

end

degree = 3     # Does not affect volume conservation
cppdegree = -1 # Does not affect volume conservation

run_case(12,degree,cppdegree)
@test run_case(48,degree,cppdegree) < 1.0e-3

end # module