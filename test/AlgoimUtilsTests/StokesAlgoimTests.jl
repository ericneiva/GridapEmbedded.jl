module StokesAgFEMTests

using Gridap
using GridapEmbedded
using GridapEmbedded.Interfaces
using Gridap.ReferenceFEs
using GridapPardiso
using SparseMatricesCSR
using Test

function main(n::Int,w::Float64,ν::Float64,γ::Float64)

  pmin = Point(-3.0,-1.0)
  pmax = Point( 3.0, 1.0)
  partition = (3*n,n)

  model = CartesianDiscreteModel(pmin,pmax,partition)
  Ω = Triangulation(model)
  dp = pmax - pmin
  h = dp[2]/n

  order = 2
  degree = order == 1 ? 3 : 2*order

  reffeᵠ = ReferenceFE(lagrangian,Float64,order)
  Vbg = FESpace(Ω,reffeᵠ)

  R = 0.12
  φ(x) = 1.0 - ( ( x[1]*x[1] + x[2]*x[2] ) / R^2 )
  φₕ = interpolate_everywhere(φ,Vbg)
  phi = AlgoimCallLevelSetFunction(φₕ,∇(φₕ))
  
  vquad = Quadrature(algoim,phi,degree,phase=IN)
  Ωᵃ,dΩ,cell_to_is_active = TriangulationAndMeasure(Ω,vquad)

  squad = Quadrature(algoim,phi,degree)
  _,dΓ,cell_to_is_cut = TriangulationAndMeasure(Ω,squad)
  n_Γ = normal(phi,Ω) # Exterior to liquid

  aggregates = aggregate(Ω,cell_to_is_active,cell_to_is_cut,IN)

  reffeᵘ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffeˢ = ReferenceFE(lagrangian,VectorValue{2,Float64},order,space=:S)
  reffeᵖ = ReferenceFE(lagrangian,Float64,order-1,space=:P)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"inlet",[1,2,3,4,7,8])
  add_tag_from_tags!(labels,"walls",[5,6])

  Vstd = TestFESpace(Ωᵃ,reffeᵘ,dirichlet_tags=["inlet"])
  Vser = TestFESpace(Ωᵃ,reffeˢ,conformity=:L2)
  Qstd = TestFESpace(Ωᵃ,reffeᵖ)

  V = AgFEMSpace(Vstd,aggregates,Vser)
  Q = AgFEMSpace(Qstd,aggregates)
  K = ConstantFESpace(model)

  uᵢ = VectorValue(w,0.0)
  u₀ = VectorValue(0.0,0.0)
  U = TrialFESpace(V,[u₀])
  P = TrialFESpace(Q)
  L = TrialFESpace(K)

  Y = MultiFieldFESpace([V,Q,K])
  X = MultiFieldFESpace([U,P,L])

  a((u,p,l),(v,q,k)) =
    ∫( ν*(ε(u)⊙ε(v)) - q*(∇⋅u) - (∇⋅v)*p )dΩ +
    ∫( (γ/h)*(u⋅v) - (v⋅(n_Γ⋅ε(u))) - ((n_Γ⋅ε(v))⋅u) + (p*n_Γ)⋅v + (q*n_Γ)⋅u )dΓ + 
    ∫( p*k )dΩ + ∫( q*l )dΩ

  l((v,q,k)) = ∫( (γ/h)*(uᵢ⋅v) - ((n_Γ⋅ε(v))⋅uᵢ) + (q*n_Γ)⋅uᵢ )dΓ

  assem = SparseMatrixAssembler(SymSparseMatrixCSR{1,Float64,Int},Vector{Float64},X,Y)
  op = AffineFEOperator(a,l,X,Y,assem)

  A = get_matrix(op)
  b = get_vector(op)
  x = similar(b)
  msglvl = 0
  # Customizing solver for real symmetric indefinite matrices
  # See https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/sparse-solver-rINines/onemkl-pardiso-parallel-direct-sparse-solver-iface/pardiso-iparm-parameter.html
  iparm = new_iparm()
  iparm[0+1] = 1 # Use default values (0 = enabled)
  iparm[1+1] = 3 # Fill-in reducing ordering for the input matrix
  iparm[9+1] = 8 # Pivoting perturbation
  iparm[10+1] = 1 # Scaling vectors
  iparm[12+1] = 1 # Improved accuracy using (non-) symmetric weighted matching
  # Note that in the analysis phase (phase=11) you must provide the numerical values 
  # of the matrix A in array a in case of scaling and symmetric weighted matching.
  iparm[17+1] = -1 # Report the number of non-zero elements in the factors. 
  iparm[18+1] = -1 # Report number of floating point operations (in 10^6 floating point operations) that are necessary to factor the matrix A.
  iparm[20+1] = 1 # Pivoting for symmetric indefinite matrices. 
  ps = PardisoSolver(GridapPardiso.MTYPE_REAL_SYMMETRIC_INDEFINITE, iparm, msglvl)
  ss = symbolic_setup(ps, A)
  ns = numerical_setup(ss, A)
  solve!(x, ns, b)
  xh = FEFunction(X,x)
  uh, ph = xh

  _A = get_matrix(op)
  _b = get_vector(op)
  _x = get_free_dof_values(xh)
  _r = _A*_x - _b
  nr = norm(_r)
  nb = norm(_b)
  nx = norm(_x)
  # @show nr, nr/nb, nr/nx
  tol_warn  = 1.0e-10
  if nr > tol_warn && nr/nb > tol_warn && nr/nx > tol_warn
    @warn "Solver not accurate"
  end

  colors = color_aggregates(aggregates,model)
  writevtk(Ω,"res_bg",celldata=["aggregate"=>aggregates,"color"=>colors])
  writevtk(Ωᵃ,"res_act",cellfields=["uh"=>uh,"ph"=>ph])
  nh = interpolate_everywhere(n_Γ,Vstd)  
  σn = ν*(ε(uh)⋅nh) - ph*nh
  writevtk(dΓ,"res_gam",cellfields=["uh"=>uh,"ph"=>ph,"sn"=>σn],qhulltype=convexhull)

end

n = 80
w = 1.0e-3
ν = 1.0e-2
γ = 1.0e-2

@info "Values: n = $n, w = $w, ν = $ν, γ = $γ"

main(n,w,ν,γ)

end # module