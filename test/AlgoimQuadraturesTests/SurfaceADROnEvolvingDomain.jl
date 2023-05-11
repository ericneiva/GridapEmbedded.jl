module SurfaceADROnEvolvingDomain

using Test
using Gridap
using Gridap.ReferenceFEs
using GridapEmbedded
using GridapEmbedded.Interfaces
using GridapEmbedded.AlgoimQuadratures

# Surface gradient
left_project(u,n) = u - n⊗(n⋅u)
directional_gradient(u,n) = left_project(∇(u),n)
∇ᵈ(u,n) = directional_gradient(u,n)

# Signed distance function (unit disk)
R = 1.0
φ(x,t) = √( (x[1]-t/√(2.0))^2 + (x[2]-t/√(2.0))^2 ) - R
φ(t::Real) = x -> φ(x,t)

# Unit normal
∇φ(x,t) = ∇(y -> φ(t)(y),x)
∇φ(t::Real) = x -> ∇φ(x,t)

# Solution
u(x,t) = cos(2*x[1])*sin(x[2])*cos(t)
u(t::Real) = x -> u(x,t)

∇u(x,t) = ∇(y -> u(t)(y),x)

∇ˢu(x,t)  = ∇u(x,t) - ∇φ(x,t)⊗(∇φ(x,t)⋅∇u(x,t))
∇ˢu(t::Real) = x -> ∇ˢu(x,t)

∇∇ˢu(x,t) = ∇(y -> ∇ˢu(t)(y),x)
∇ˢ∇ˢu(x,t) = ∇∇ˢu(x,t) - ∇φ(x,t)⊗(∇φ(x,t)⋅∇∇ˢu(x,t))

Δˢu(x,t) = tr(∇ˢ∇ˢu(x,t))

# Advection field
b = VectorValue(1.0,1.0)/√(2.0)

# Source term
f(x,t) = ∂t(u)(t)(x) + b⋅∇ˢu(x,t) - Δˢu(x,t) # (∇ˢ⋅b) u = 0
f(t::Real) = x -> f(x,t)

function run_trace_fem(nx::Int,ny::Int)

  domain = (-1.0,2.0,-1.0,2.0)
  cells = (nx,ny)
  h = (domain[2]-domain[1])/cells[1]
  model = CartesianDiscreteModel(domain,cells)
  Ω = Triangulation(model)

  order = 1
  degree = order == 1 ? 3 : 2*order

  buffer = Ref{Any}((Ω=nothing,dΩbg=nothing,dΓ=nothing,t=nothing))

  function update_buffer!(t,dt)
    if buffer[].t == t
      return true
    else

      # Next time volume quadrature
      φ₊ = AlgoimCallLevelSetFunction(φ(t+dt),∇φ(t+dt))
      squad = Quadrature(algoim,φ₊,degree,phase=CUT)
      dΩbg₊ = Measure(Ω,squad,data_domain_style=PhysicalDomain())

      # Current time volume quadrature
      φ₋ = AlgoimCallLevelSetFunction(φ(t+dt),∇φ(t+dt))
      if buffer[].dΩbg === nothing
        squad = Quadrature(algoim,φ₋,degree,phase=CUT)
        dΩbg₋ = Measure(Ω,squad,data_domain_style=PhysicalDomain())
      else
        dΩbg₋ = buffer[].dΩbg
      end

      # Narrow band active triangulation     
      is_a₋ = is_cell_active(dΩbg₋)
      is_a₊ = is_cell_active(dΩbg₊)
      is_n = lazy_map((a₋,a₊)->a₋|a₊,is_a₋,is_a₊)
      Ωₜ = Triangulation(Ω,is_n)

      # Current volume measure at narrow band
      dΩₜ = Measure(Ωₜ,2*order)

      # Current time surface measure
      dΓₜ = restrict_measure(dΩbg₋,Triangulation(Ω,is_a₋))

      # Update buffer
      buffer[] = (Ω=Ωₜ,dΩ=dΩₜ,dΓ=dΓₜ,dΩbg=dΩbg₊,t=t)
      return true

    end
  end

  dΓ() = buffer[].dΓ
  dΩ() = buffer[].dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)

  function update_fe_spaces(t::Real,dt::Real)
    
    update_buffer!(t,dt)
    Ωₜ = buffer[].Ω
    
    V = TestFESpace(Ωₜ,reffe)
    U = TransientTrialFESpace(V)

    U,V

  end

  a(t,u,v) = ∫( ∇ᵈ(u,∇φ(t))⋅∇ᵈ(v,∇φ(t)) )dΓ() + ∫( (b⋅∇ᵈ(u,∇φ(t)))*v )dΓ() # (∇ˢ⋅b) u = 0

  # Γ-orthogonal derivative volume stabilisation
  γˢ = 2.0
  μˢ = 1.0
  ηˢ = μˢ/h + (γˢ/h)*(1.0+1.0/h) # dt = h
  s(t,u,v) = ∫( ηˢ*((∇φ(t)⋅∇(u))⊙(∇φ(t)⋅∇(v))) )dΩ()

  M(t,ut,v) = ∫( ut*v )dΓ()
  A(t,u,v) = a(t,u,v) + s(t,u,v)
  B(t,v) = ∫( f(t)*v )dΓ()

  t0 = 0.0
  tF = 1.0
  dt = h

  ls = LUSolver()
  ode_solver = ThetaMethod(ls,dt,1.0)

  l2(w) = w⋅w
  h1(w,t) = ∇ᵈ(w,∇φ(t))⊙∇ᵈ(w,∇φ(t))
  
  el2 = 0.0; eh1 = 0.0;
  l2u = 0.0; h1u = 0.0;

  U,V = update_fe_spaces(t0,dt)
  uᵢ = interpolate_everywhere(u(t0),U(t0))

  println("Begin temporal loop")

  for ti in t0:dt:(tF-dt)

    t = ti+dt
    U,V = update_fe_spaces(t,dt)
    uᵢ = interpolate_everywhere(uᵢ,U(t))

    op = TransientAffineFEOperator(M,A,B,U,V)

    uₕₜ = solve(ode_solver,op,uᵢ,ti,t)
    (uᵢ,t),_ = Base.iterate(uₕₜ)
    println("Time: $t")

    e = u(t) - uᵢ
    w = CellField(u(t),Ω)

    sx = dΓ().quad.cell_point.values
    sx = lazy_map(Reindex(sx),dΓ().quad.cell_point.ptrs)
    usx = lazy_map(uᵢ,sx)
    esx = lazy_map(e,sx)
    writevtk(sx,"res_$t",nodaldata=["uₕ"=>usx,"eₕ"=>esx])

    _el2 = ∑( ∫( l2(e) )dΓ() )
    _eh1 = ∑( ∫( h1(e,t) )dΓ() )

    @show _el2
    @show _eh1

    el2 = el2 + _el2
    l2u = l2u + ∑( ∫( l2(w) )dΓ() )
    eh1 = eh1 + _eh1
    h1u = h1u + ∑( ∫( h1(w,t) )dΓ() )

  end

  el2 = √(dt*el2) / √(dt*l2u)
  eh1 = √(dt*eh1) / √(dt*h1u)

  @show el2
  @show eh1

end

run_trace_fem(30,30)
run_trace_fem(60,60)

end # module