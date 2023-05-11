module MeanCurvatureAlgoimTests

using Test
using Gridap
using Gridap.ReferenceFEs
using GridapEmbedded
using GridapEmbedded.Interfaces

R = 1.0
r = 0.5

φ(x) = √( x[3]^2 + ( √( x[1]^2 + x[2]^2 ) - R )^2 ) - r # Torus LS
∇φ(x) = ∇(φ)(x)
Δφ(x) = Δ(φ)(x)

phi = AlgoimCallLevelSetFunction(φ,∇φ)

# Projection operator
p = VectorValue(0.0,0.0,0.0)
id = one(∇φ(p)⊗∇φ(p))
P(x) = id - ∇φ(x)⊗∇φ(x)

# Mean curvature vector for torus
H(x) = Δφ(x)*∇φ(x)

left_project(u,n) = u - n⊗(n⋅u)
directional_gradient(u,n) = left_project(∇(u),n)
∇ᵈ(u,n) = directional_gradient(u,n)

function run_mean_curvature(domain,cells,order)

  model = CartesianDiscreteModel(domain,cells)
  Ω = Triangulation(model)

  degree = order == 1 ? 3 : 2*order
  squad = Quadrature(algoim,phi,degree)
  Ωᶜ,dΓ = TriangulationAndMeasure(Ω,squad)

  n_Γ = normal(phi,Ωᶜ)
  dΩᶜ = Measure(Ωᶜ,2*order)

  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  V = TestFESpace(Ωᶜ,reffe)
  U = TrialFESpace(V)

  m(u,v) = ∫( u⋅v )dΓ

  l(v) = ∫( P⊙∇ᵈ(v,n_Γ) )dΓ

  # Γ-orthogonal derivative volume stabilisation
  h = (domain[2]-domain[1])/cells[1]
  γˢ = 10.0*h
  s(u,v) = ∫( γˢ*((n_Γ⋅∇(u))⊙(n_Γ⋅∇(v))) )dΩᶜ

  a(u,v) = m(u,v) + s(u,v)

  # FE problem
  op = AffineFEOperator(a,l,U,V)
  uₕ = solve(op)

  eₕ = H - uₕ

  l2(u) = √(∑( ∫( u⋅u )dΓ ))
  h1(u) = √(∑( ∫( u⋅u + ∇ᵈ(u,n_Γ)⊙∇ᵈ(u,n_Γ) )dΓ ))

  el2 = l2(eₕ)
  eh1 = h1(eₕ)

  @show el2, eh1
  
  # x = dΓ.quad.cell_point.values
  # x = lazy_map(Reindex(x),dΓ.quad.cell_point.ptrs)
  # ux = lazy_map(uₕ,x)
  # ex = lazy_map(eₕ,x)
  # writevtk(x,"mc",nodaldata=["uₕ"=>ux,"eₕ"=>ex])

end

run_mean_curvature((-1.6,1.6,-1.6,1.6,-1.6,1.6),(11,11,11),2)
run_mean_curvature((-1.6,1.6,-1.6,1.6,-1.6,1.6),(22,22,22),2)
run_mean_curvature((-1.6,1.6,-1.6,1.6,-1.6,1.6),(44,44,44),2)
run_mean_curvature((-1.6,1.6,-1.6,1.6,-1.6,1.6),(88,88,88),2)

end # module