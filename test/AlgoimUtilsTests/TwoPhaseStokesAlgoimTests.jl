module TwoPhaseStokesAgFEMTests

using Gridap
using GridapEmbedded
using GridapEmbedded.Interfaces
using Gridap.ReferenceFEs
using GridapPardiso
using SparseMatricesCSR
using Test

function steady_state(n::Int,w::Float64,νˡ::Float64,νˢ::Float64,γ::Float64)

  pmin = Point(-1.5,-1.0)
  pmax = Point( 4.5, 1.0)
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
  
  lquad = Quadrature(algoim,phi,degree,phase=IN)
  Ωˡ,dΩˡ,cell_to_is_liquid = TriangulationAndMeasure(Ω,lquad)
  
  squad = Quadrature(algoim,phi,degree,phase=OUT)
  Ωˢ,dΩˢ,cell_to_is_solid = TriangulationAndMeasure(Ω,squad)

  iquad = Quadrature(algoim,phi,degree)
  _,dΓ,cell_to_is_cut = TriangulationAndMeasure(Ω,iquad)
  n_Γ = normal(phi,Ω) # Exterior to liquid

  aggsˡ = aggregate(Ω,cell_to_is_liquid,cell_to_is_cut,IN)
  aggsˢ = aggregate(Ω,cell_to_is_solid,cell_to_is_cut,OUT)

  reffeᵘ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffeˢ = ReferenceFE(lagrangian,VectorValue{2,Float64},order,space=:S)
  reffeᵖ = ReferenceFE(lagrangian,Float64,order-1,space=:P)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"inlet",[7])

  Vˡstd = TestFESpace(Ωˡ,reffeᵘ,dirichlet_tags=["inlet"])
  Vˡser = TestFESpace(Ωˡ,reffeˢ,conformity=:L2)
  Qˡstd = TestFESpace(Ωˡ,reffeᵖ)

  Vˢstd = TestFESpace(Ωˢ,reffeᵘ)
  Vˢser = TestFESpace(Ωˢ,reffeˢ,conformity=:L2)
  Qˢstd = TestFESpace(Ωˢ,reffeᵖ)

  Vˡ = AgFEMSpace(Vˡstd,aggsˡ,Vˡser)
  Qˡ = AgFEMSpace(Qˡstd,aggsˡ)
  Vˢ = AgFEMSpace(Vˢstd,aggsˢ,Vˢser)
  Qˢ = AgFEMSpace(Qˢstd,aggsˢ)  
  K = ConstantFESpace(model)

  uᵢ(x) = VectorValue(w*(1.0-x[2]*x[2]),0.0)
  Uˡ = TrialFESpace(Vˡ,[uᵢ])
  Pˡ = TrialFESpace(Qˡ)
  Uˢ = TrialFESpace(Vˢ)
  Pˢ = TrialFESpace(Qˢ)
  L = TrialFESpace(K)

  Y = MultiFieldFESpace([Vˡ,Qˡ,K,Vˢ,Qˢ,K])
  X = MultiFieldFESpace([Uˡ,Pˡ,L,Uˢ,Pˢ,L])

  wˡ = CellField(νˢ/(νˡ+νˢ),Ω)
  wˢ = CellField(νˡ/(νˡ+νˢ),Ω)
  νᵞ = 2*νˡ*νˢ/(νˡ+νˢ)

  jumpᵘ(uˡ,uˢ) = uˡ-uˢ
  σ(ε,ν) = 2*ν*ε
  τ(ε) = one(ε)
  meanᵗ(uˡ,pˡ,νˡ,uˢ,pˢ,νˢ) = 
    wˡ*(σ(ε(uˡ),νˡ))-pˡ*(τ∘(ε(uˡ))) + 
    wˢ*(σ(ε(uˢ),νˢ))-pˢ*(τ∘(ε(uˢ)))

  a((uˡ,pˡ,lˡ,uˢ,pˢ,lˢ),(vˡ,qˡ,kˡ,vˢ,qˢ,kˢ)) =
    ∫( 2*νˡ*(ε(uˡ)⊙ε(vˡ)) - qˡ*(∇⋅uˡ) - (∇⋅vˡ)*pˡ )dΩˡ +
    ∫( 2*νˢ*(ε(uˢ)⊙ε(vˢ)) - qˢ*(∇⋅uˢ) - (∇⋅vˢ)*pˢ )dΩˢ +
    ∫( pˡ*kˡ )dΩˡ + ∫( qˡ*lˡ )dΩˡ +
    ∫( pˢ*kˢ )dΩˢ + ∫( qˢ*lˢ )dΩˢ +
    ∫( (γ*νᵞ/h)*(jumpᵘ(uˡ,uˢ)⋅jumpᵘ(vˡ,vˢ)) - 
       (jumpᵘ(vˡ,vˢ)⋅(n_Γ⋅meanᵗ(uˡ,pˡ,νˡ,uˢ,pˢ,νˢ))) - 
       ((n_Γ⋅meanᵗ(vˡ,qˡ,νˡ,vˢ,qˢ,νˢ))⋅jumpᵘ(uˡ,uˢ)) )dΓ

  l((vˡ,qˡ,kˡ,vˢ,qˢ,kˢ)) = 0.0

  assem = SparseMatrixAssembler(SymSparseMatrixCSR{1,Float64,Int},Vector{Float64},X,Y)
  op = AffineFEOperator(a,l,X,Y,assem)

  A = get_matrix(op)
  b = get_vector(op)
  x = similar(b)
  msglvl = 0
  # Customizing solver for real symmetric indefinite matrices
  # See https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface/pardiso-iparm-parameter.html
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
  uhl, phl, _, uhs, phs, _ = xh

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

  colors = color_aggregates(aggsˡ,model)
  writevtk(Ω,"res_bg_l",celldata=["aggregate"=>aggsˡ,"color"=>colors])
  writevtk(Ω,"res_bg_s",celldata=["aggregate"=>aggsˢ,"color"=>colors])
  writevtk(Ωˡ,"res_l",cellfields=["uhl"=>uhl,"phl"=>phl])
  writevtk(dΩˢ,"res_s",cellfields=["uhs"=>uhs,"phs"=>phs])
  writevtk(dΓ,"res_gam",cellfields=["uhl"=>uhl,"uhs"=>uhs],qhulltype=convexhull)
  # nh = interpolate_everywhere(n_Γ,Vˡstd)  
  # σn = νˡ*(ε(uhl)⋅nh) # - phl*nh
  # writevtk(dΓ,"res_gam",cellfields=["uhl"=>uhl,"phl"=>phl,"sn"=>σn],qhulltype=convexhull)

end

function dynamic(n::Int,w::Float64,
                 νˡ::Float64,νˢ::Float64,γ::Float64,
                 t₀::Float64,Δt₀::Float64,T::Float64)

  pmin = Point(-1.5,-2.5)
  pmax = Point( 3.5, 2.5)
  partition = (n,n)

  model = CartesianDiscreteModel(pmin,pmax,partition)
  Ω = Triangulation(model)
  dp = pmax - pmin
  h = dp[2]/n

  order = 2
  degree = order == 1 ? 3 : 2*order

  R = 0.12
  c₀ = Point(-1.0,1.0)

  # Buffer of active model and integration objects
  buffer = Ref{Any}((Ωˡ=nothing,Ωˢ=nothing,
                     dΩˡ=nothing,dΩˢ=nothing,
                     dΓ=nothing,n_Γ=nothing,
                     aggsˡ=nothing,aggsˢ=nothing,
                     φ₋=nothing,c₋=nothing,t=nothing))

  function update_buffer!(t,dt,v₋₂)

    if buffer[].t == t
      return true
    else

      if buffer[].c₋ === nothing
        c₋ = c₀
      else
        c₋ = buffer[].c₋
        c₋ = c₋ + dt * v₋₂
      end

      φ(x) = 1.0 - ( ( (x[1]-c₋[1])*(x[1]-c₋[1]) + 
                       (x[2]-c₋[2])*(x[2]-c₋[2]) ) / R^2 )
      φ₋ = AlgoimCallLevelSetFunction(φ,∇(φ))

      lquad = Quadrature(algoim,φ₋,degree,phase=IN)
      Ωˡ,dΩˡ,cell_to_is_liquid = TriangulationAndMeasure(Ω,lquad)
      
      squad = Quadrature(algoim,φ₋,degree,phase=OUT)
      Ωˢ,dΩˢ,cell_to_is_solid = TriangulationAndMeasure(Ω,squad)
    
      iquad = Quadrature(algoim,φ₋,degree)
      _,dΓ,cell_to_is_cut = TriangulationAndMeasure(Ω,iquad)
      n_Γ = normal(φ₋,Ω) # Exterior to liquid
    
      aggsˡ = aggregate(Ω,cell_to_is_liquid,cell_to_is_cut,IN)
      aggsˢ = aggregate(Ω,cell_to_is_solid,cell_to_is_cut,OUT)

      buffer[] = (Ωˡ=Ωˡ,Ωˢ=Ωˢ,dΩˡ=dΩˡ,dΩˢ=dΩˢ,
                  dΓ=dΓ,n_Γ=n_Γ,aggsˡ=aggsˡ,aggsˢ=aggsˢ,
                  c₋=c₋,t=t)
      return true

    end

  end

  reffeᵘ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffeˢ = ReferenceFE(lagrangian,VectorValue{2,Float64},order,space=:S)
  reffeᵖ = ReferenceFE(lagrangian,Float64,order-1,space=:P)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"inlet",[7])

  function update_all!(t::Real,dt::Real,disp)

    update_buffer!(t,dt,disp)

    Ωˡ  = buffer[].Ωˡ
    Ωˢ  = buffer[].Ωˢ
 
    dΩˡ = buffer[].dΩˡ
    dΩˢ = buffer[].dΩˢ
 
    dΓ  = buffer[].dΓ
    n_Γ = buffer[].n_Γ

    aggsˡ = buffer[].aggsˡ
    aggsˢ = buffer[].aggsˢ

    cₜ = buffer[].c₋

    Vˡstd = TestFESpace(Ωˡ,reffeᵘ,dirichlet_tags=["inlet"])
    Vˡser = TestFESpace(Ωˡ,reffeˢ,conformity=:L2)
    Qˡstd = TestFESpace(Ωˡ,reffeᵖ)
  
    Vˢstd = TestFESpace(Ωˢ,reffeᵘ)
    Vˢser = TestFESpace(Ωˢ,reffeˢ,conformity=:L2)
    Qˢstd = TestFESpace(Ωˢ,reffeᵖ)
  
    Vˡ = AgFEMSpace(Vˡstd,aggsˡ,Vˡser)
    Qˡ = AgFEMSpace(Qˡstd,aggsˡ)
    Vˢ = AgFEMSpace(Vˢstd,aggsˢ,Vˢser)
    Qˢ = AgFEMSpace(Qˢstd,aggsˢ)  
    K = ConstantFESpace(model)
  
    uᵢ(x) = VectorValue(w*(6.25-x[2]*x[2]),0.0)
    Uˡ = TrialFESpace(Vˡ,[uᵢ])
    Pˡ = TrialFESpace(Qˡ)
    Uˢ = TrialFESpace(Vˢ)
    Pˢ = TrialFESpace(Qˢ)
    L = TrialFESpace(K)
  
    Y = MultiFieldFESpace([Vˡ,Qˡ,K,Vˢ,Qˢ,K])
    X = MultiFieldFESpace([Uˡ,Pˡ,L,Uˢ,Pˢ,L])

    X,Y,dΩˡ,dΩˢ,Ωˡ,Ωˢ,dΓ,n_Γ,cₜ

  end

  Δt = Δt₀

  X,Y,dΩˡ,dΩˢ,Ωˡ,Ωˢ,dΓ,n_Γ,cₜ = update_all!(t₀,Δt,VectorValue(0.0,0.0))

  wˡ = CellField(νˢ/(νˡ+νˢ),Ω)
  wˢ = CellField(νˡ/(νˡ+νˢ),Ω)
  νᵞ = 2*νˡ*νˢ/(νˡ+νˢ)

  jumpᵘ(uˡ,uˢ) = uˡ-uˢ
  σ(ε,ν) = 2*ν*ε
  τ(ε) = one(ε)
  meanᵗ(uˡ,pˡ,νˡ,uˢ,pˢ,νˢ) = 
    wˡ*(σ(ε(uˡ),νˡ))-pˡ*(τ∘(ε(uˡ))) + 
    wˢ*(σ(ε(uˢ),νˢ))-pˢ*(τ∘(ε(uˢ)))

  a((uˡ,pˡ,lˡ,uˢ,pˢ,lˢ),(vˡ,qˡ,kˡ,vˢ,qˢ,kˢ)) =
    ∫( 2*νˡ*(ε(uˡ)⊙ε(vˡ)) - qˡ*(∇⋅uˡ) - (∇⋅vˡ)*pˡ )dΩˡ +
    ∫( 2*νˢ*(ε(uˢ)⊙ε(vˢ)) - qˢ*(∇⋅uˢ) - (∇⋅vˢ)*pˢ )dΩˢ +
    ∫( pˡ*kˡ )dΩˡ + ∫( qˡ*lˡ )dΩˡ +
    ∫( pˢ*kˢ )dΩˢ + ∫( qˢ*lˢ )dΩˢ +
    ∫( (γ*νᵞ/h)*(jumpᵘ(uˡ,uˢ)⋅jumpᵘ(vˡ,vˢ)) - 
       (jumpᵘ(vˡ,vˢ)⋅(n_Γ⋅meanᵗ(uˡ,pˡ,νˡ,uˢ,pˢ,νˢ))) - 
       ((n_Γ⋅meanᵗ(vˡ,qˡ,νˡ,vˢ,qˢ,νˢ))⋅jumpᵘ(uˡ,uˢ)) )dΓ

  l((vˡ,qˡ,kˡ,vˢ,qˢ,kˢ)) = 0.0

  assem = SparseMatrixAssembler(SymSparseMatrixCSR{1,Float64,Int},Vector{Float64},X,Y)
  op = AffineFEOperator(a,l,X,Y,assem)

  A = get_matrix(op)
  b = get_vector(op)
  x = similar(b)
  msglvl = 0
  # Customizing solver for real symmetric indefinite matrices
  # See https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface/pardiso-iparm-parameter.html
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
  uhl, phl, _, uhs, phs, _ = xh

  # _A = get_matrix(op)
  # _b = get_vector(op)
  # _x = get_free_dof_values(xh)
  # _r = _A*_x - _b
  # nr = norm(_r)
  # nb = norm(_b)
  # nx = norm(_x)
  # # @show nr, nr/nb, nr/nx
  # tol_warn  = 1.0e-10
  # if nr > tol_warn && nr/nb > tol_warn && nr/nx > tol_warn
  #   @warn "Solver not accurate"
  # end

  i = 0
  t = t₀

  writevtk(Ωˡ,"res_l_$i",cellfields=["uhl"=>uhl,"phl"=>phl])
  writevtk([dΩˢ,dΓ],"res_s_$i",cellfields=["uhs"=>uhs,"phs"=>phs])
  writevtk(dΓ,"res_gam_$i",cellfields=["uhl"=>uhl,"uhs"=>uhs],qhulltype=convexhull)

  while t < T

    i = i + 1
    t = t + Δt

    @info "Time step $i, time $t and time step $Δt"
  
    ω = ( ∑( ∫( uhs )dΓ ) / ∑( ∫( 1.0 )dΓ ) )
    @show ω,uhs(cₜ),norm(ω-uhs(cₜ))
    @assert norm(ω-uhs(cₜ)) < 1.0e-03

    X,Y,dΩˡ,dΩˢ,Ωˡ,Ωˢ,dΓ,n_Γ,cₜ = update_all!(t,Δt,ω)

    assem = SparseMatrixAssembler(SymSparseMatrixCSR{1,Float64,Int},Vector{Float64},X,Y)
    op = AffineFEOperator(a,l,X,Y,assem)

    A = get_matrix(op)
    b = get_vector(op)
    x = similar(b)

    ss = symbolic_setup(ps, A)
    ns = numerical_setup(ss, A)
    solve!(x, ns, b)
    xh = FEFunction(X,x)
    uhl, phl, _, uhs, phs, _ = xh

    writevtk(Ωˡ,"res_l_$i",cellfields=["uhl"=>uhl,"phl"=>phl])
    writevtk([dΩˢ,dΓ],"res_s_$i",cellfields=["uhs"=>uhs,"phs"=>phs])
    writevtk(dΓ,"res_gam_$i",cellfields=["uhl"=>uhl,"uhs"=>uhs],qhulltype=convexhull)

  end

end

n  = 80
w  = 1.0e-1
νˡ = 1.0e-1
νˢ = 1.0e+2
γ  = 1.0e-1 # Scale with νˢ
t₀ = 0.0
Δt = 0.1
T  = 3.0

@info "Values: n = $n, w = $w, νˡ = $νˡ, νˢ = $νˢ, γ = $γ"

dynamic(n,w,νˡ,νˢ,γ,t₀,Δt,T)

end # module