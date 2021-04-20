module ModalC0AgFEMTests

  using DrWatson
  @quickactivate "GridapEmbedded"

  using Gridap
  using GridapEmbedded
  using GridapEmbedded.AgFEM

  using GridapEmbedded.Interfaces: CUT, IN

  using LinearAlgebra: cond
  using SparseArrays: SparseMatrixCSC

  using GridapPETSc
  using MPI

  include("StaticCondensation.jl")

  const tol = 1e-16
  const maxits = 5000
  const R = 0.42

  function compute(n::Int,k::Int,d::Int,t::Int,s::Int,g::Int)

    u(x) = (1-s)*sum(x)+s*(sum(x)^(k+1))
    w(x) = -Δ(u)(x)
    ud(x) = u(x)

    if d == 2
      if g == 0
        geom = disk(R,x0=Point(0.5,0.5))
      else
        geom = square(L=0.63,x0=Point(0.5,0.5))
      end
      partition = (n,n)
      domain = (0,1,0,1)
    else
      if g == 0
        geom = sphere(R,x0=Point(0.5,0.5,0.5))
      else
        geom = doughnut(0.21,0.1,x0=Point(0.5,0.5,0.5))
      end
      partition = (n,n,n)
      domain = (0,1,0,1,0,1)
    end

    bgmodel = CartesianDiscreteModel(domain,partition)
    h = (domain[2]-domain[1])/n
    cutgeo = cut(bgmodel,geom)
    model = DiscreteModel(cutgeo)

    if t == 0
      γ₀ = 0.65
      eᵧ = (3-2/num_dims(model))/(2*k+1-2/num_dims(model))
      γ = γ₀^eᵧ
      strategy = AggregateCutCellsByThreshold(γ)
    else
      strategy = AggregateAllCutCells()
    end

    aggregates = aggregate(strategy,cutgeo,geom)
    bboxes = compute_cell_to_dface_bboxes(model,aggregates)

    Ω_bg = Triangulation(bgmodel)
    amodel = DiscreteModel(cutgeo,geom,(CUT,IN))
    Ω = Triangulation(amodel)
    Γ = EmbeddedBoundary(cutgeo)

    n_Γ = get_normal_vector(Γ)

    D = num_dims(model)
    if s == 0
      cdeg, degw, degh, dege = 2*k*D, max(2*k-1,2), max(2*k-2,2), 2*k # Or degw = 2*k-2
    else
      cdeg, degw, degh, dege = (2*k+1)*D, max(2*k-1,2), 2*k, 2*k+2
    end
    dΩ = Measure(MomentFittingQuad(Ω,cutgeo,degw))
    dH = Measure(MomentFittingQuad(Ω,cutgeo,degh))
    dO = Measure(MomentFittingQuad(Ω,cutgeo,dege))
    dΓ = Measure(Γ,cdeg)

    γd = 5.0*k^2

    a(u,v) =
      ∫( ∇(u)⋅∇(v) )dΩ +
      ∫( (γd/h)*u*v  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ

    l(v) =
      ∫( v*w )dΩ +
      ∫( (γd/h)*v*ud - (n_Γ⋅∇(v))*ud )dΓ

    l2(u) = sqrt(abs(sum(∫( u*u )dO)))
    h1(u) = sqrt(abs(sum(∫( ∇(u)⋅∇(u) )dH)))

    reffe = ReferenceFE(modalC0,Float64,k,bboxes)
    Vstd = TestFESpace(model,reffe,conformity=:H1)
    V = AgFEMSpace(Vstd,aggregates,modalC0)
    U = TrialFESpace(V)

    cell_matvec, bface_matvec = compute_contributions(U,V,a,l,Ω,Γ)
    cell_matvec, ccell_to_bgcell =
      combine_cell_and_bface_contribs(Ω_bg,Ω,Γ,cell_matvec,bface_matvec)
    cell_matvec = attach_constraints_cols(U,cell_matvec,ccell_to_bgcell)
    cell_matvec = attach_constraints_rows(V,cell_matvec,ccell_to_bgcell)

    cell_dofs = lazy_map(Reindex(get_cell_dof_ids(V)),ccell_to_bgcell)
    idof_to_dof, dof_to_idof = compute_itfc_dofs(cell_dofs,V)
    cell_ildof_ldof, cell_cldof_ldof, cell_idofs =
      compute_itfc_cell_dof_arrays(cell_dofs,dof_to_idof)

    dict = Dict()
    cell_imatvec = lazy_map(StaticCondensationMap(dict),cell_matvec,cell_ildof_ldof,cell_cldof_ldof)

    sop = @timed S,f = assemble_schur_system(cell_imatvec,cell_idofs,idof_to_dof)
    if k < 3
      scn = @timed kopM = cond(S,1)
    else
      scn = @timed kopM = 0
    end

    solver = PETScSolver()
    sso = @timed x = solve(solver,S,f)
    iteM = PETSc_get_number_of_iterations(solver)

    y = apply_interior_correction(x,V,
                                  idof_to_dof,dof_to_idof,cell_matvec,
                                  cell_dofs,cell_ildof_ldof,cell_cldof_ldof,
                                  dict)
    uh = FEFunction(V,y)

    e = u - uh
    el2M = l2(e)
    eh1M = h1(e)

    if k != 1
      reffe = ReferenceFE(lagrangian,Float64,k)
      Vstd = TestFESpace(model,reffe,conformity=:H1)
      V = AgFEMSpace(Vstd,aggregates,lagrangian)
      U = TrialFESpace(V)

      cell_matvec, bface_matvec = compute_contributions(U,V,a,l,Ω,Γ)
      cell_matvec, ccell_to_bgcell =
        combine_cell_and_bface_contribs(Ω_bg,Ω,Γ,cell_matvec,bface_matvec)
      cell_matvec = attach_constraints_cols(U,cell_matvec,ccell_to_bgcell)
      cell_matvec = attach_constraints_rows(V,cell_matvec,ccell_to_bgcell)

      cell_dofs = lazy_map(Reindex(get_cell_dof_ids(V)),ccell_to_bgcell)
      idof_to_dof, dof_to_idof = compute_itfc_dofs(cell_dofs,V)
      cell_ildof_ldof, cell_cldof_ldof, cell_idofs =
        compute_itfc_cell_dof_arrays(cell_dofs,dof_to_idof)

      dict = Dict()
      cell_imatvec = lazy_map(StaticCondensationMap(dict),cell_matvec,cell_ildof_ldof,cell_cldof_ldof)

      S,f = assemble_schur_system(cell_imatvec,cell_idofs,idof_to_dof)
      if k < 3
        scn = @timed kopN = cond(S,1)
      else
        scn = @timed kopN = 0
      end

      solver = PETScSolver()
      x = solve(solver,S,f)
      iteN = PETSc_get_number_of_iterations(solver)

      y = apply_interior_correction(x,V,
                                    idof_to_dof,dof_to_idof,cell_matvec,
                                    cell_dofs,cell_ildof_ldof,cell_cldof_ldof,
                                    dict)
      uh = FEFunction(V,y)

      e = u - uh
      el2N = l2(e)
      eh1N = h1(e)
    else
      el2N = el2M
      eh1N = eh1M
      kopN = kopM
      iteN = iteM
    end

    num_free_dofs(U)^(1/D), el2M, eh1M, kopM, iteM, el2N, eh1N, kopN, iteN, sop.time, scn.time, sso.time, sop.bytes, scn.bytes, sso.bytes

  end

  function compute(case::Dict)
    @unpack n,k,d,t,s,g = case
    udofs, el2M, eh1M, kopM, iteM, el2N, eh1N, kopN, iteN, top, tcn, tso, bop, bcn, bso = compute(n,k,d,t,s,g)
    results = @dict udofs el2M eh1M kopM iteM el2N eh1N kopN iteN top tcn tso bop bcn bso
    merge(case,results)
  end

  function compute_and_save(case::Dict)
    produce_or_load(
      datadir(),
      case,
      compute,
      prefix="modalC0",
      tag=true,
      verbose=true,
      suffix="bson"
    )
    return true
  end

  function compute()

    if !MPI.Initialized()
      MPI.Init()
    end

    GridapPETSc.Init(["-ksp_type","cg",
                      "-ksp_converged_reason",
                      "-ksp_rtol","$tol",
                      "-ksp_max_it","$maxits",
                      "-ksp_norm_type","unpreconditioned",
                      "-pc_type","gamg",
                      "-pc_gamg_type","agg",
                      "-mg_levels_esteig_ksp_type","cg",
                      "-mg_coarse_sub_pc_type","cholesky",
                      "-mg_coarse_sub_pc_factor_mat_ordering_type","nd",
                      "-pc_gamg_process_eq_limit","50",
                      "-pc_gamg_square_graph","0",
                      "-pc_gamg_agg_nsmooths","1"])

    @info "Training"
    compute(6,3,2,1,1,0)
    @info "Producing"
    params = Dict(
      :n => [6,12,24],
      :k => [1,2,3],
      :d => [2],
      :t => [0,1],
      :s => [0,1],
      :g => [0,1]
    )
    dicts = dict_list(params)
    map(compute_and_save,dicts)

    GridapPETSc.Finalize()

    # if MPI.Initialized() & !MPI.Finalized()
    #   MPI.Finalize()
    # end

  end

  # compute()
  export tol, maxits
  export compute, compute_and_save

end
