module AlgoimUtils

using CxxWrap
using Algoim
using Gridap

import Base.prod
@inline prod(x::Point{N,T}) where {N,T} = prod(x.data)

import Algoim: to_array
@inline to_array(x::Point{N,T}) where {N,T} = collect(x.data)

using Algoim: lsbuffer, lsbufferφ, lsbuffer∇φ

using Gridap.Helpers
using Gridap.ReferenceFEs
using Gridap.Arrays: collect1d, CompressedArray
using Gridap.Geometry: get_cartesian_descriptor, Grid, get_reffes
using Gridap.Adaptivity
using Gridap.Fields: testitem
using Gridap.CellData
using Gridap.CellData: GenericCellField, get_data
using Gridap.CellData: _point_to_cell_cache, _point_to_cell!
using Gridap.Geometry: get_faces, get_grid_topology

using GridapEmbedded.Interfaces
using GridapEmbedded.AgFEM: _aggregate_by_threshold_barrier

import Algoim: AlgoimCallLevelSetFunction
import Algoim: normal
import Gridap.ReferenceFEs: Quadrature
import GridapEmbedded.AgFEM: aggregate

export TriangulationAndMeasure
export algoim
export Quadrature
export is_cell_active
export restrict_measure
export aggregate_narrow_band

struct Algoim <: QuadratureName end
const algoim = Algoim()

function AlgoimCallLevelSetFunction(φ::CellField,∇φ::CellField)
  φtrian = get_triangulation(φ)
  ∇φtrian = get_triangulation(∇φ)
  xφ = testitem(testitem(get_cell_coordinates(φtrian)))
  x∇φ = testitem(testitem(get_cell_coordinates(∇φtrian)))
  cellφ = get_array(φ)
  cell∇φ = get_array(∇φ)
  cache_φ = (cellφ,array_cache(cellφ),return_cache(testitem(cellφ),xφ))
  cache_∇φ = (cell∇φ,array_cache(cell∇φ),return_cache(testitem(cell∇φ),x∇φ))
  update_lsbuffer!(cache_φ,cache_∇φ)
  AlgoimCallLevelSetFunction{typeof(φ),typeof(∇φ),typeof(cache_φ),typeof(cache_∇φ)}(φ,∇φ,cache_φ,cache_∇φ)
end

function update_lsbuffer!(cache_φ::Tuple,cache_∇φ::Tuple) 
  cppφ(p,i::Float32) = begin
    _p = Point(to_const_array(p))
    (carr,cgetindex,ceval) = cache_φ
    evaluate!(ceval,getindex!(cgetindex,carr,Int(i)),_p)
  end
  cpp∇φ(p,i::Float32) = begin
    _p = Point(to_const_array(p))
    (carr,cgetindex,ceval) = cache_∇φ
    _val = evaluate!(ceval,getindex!(cgetindex,carr,Int(i)),_p)
    ConstCxxRef(to_uvector(to_array(_val)))
  end
  lsbuffer[] = (φ=cppφ, ∇φ=cpp∇φ)
end

function normal(phi::AlgoimCallLevelSetFunction,x::AbstractVector{<:Point},cell_id::Int=1)
  map(xi->normal(phi,xi,cell_id),x)
end

function normal(phi::AlgoimCallLevelSetFunction,trian::Triangulation)
  f = [ x -> normal(phi,x,i) for i in 1:num_cells(trian) ]
  GenericCellField(f,trian,PhysicalDomain())
end

function normal(ls::AlgoimCallLevelSetFunction{<:CellField,<:CellField},x::Point,cell_id::Int=1)
  (carr,cgetindex,ceval) = ls.cache_∇φ
  gx = evaluate!(ceval,getindex!(cgetindex,carr,cell_id),x)
  gx/norm(gx)
end

function Quadrature(trian::Grid,::Algoim,args...;kwargs...)
  Quadrature(Val{num_dims(trian)}(),trian,algoim,args...;kwargs...)
end

function Quadrature(::Val{2},trian,::Algoim,args...;kwargs...)
  ctype_polytope = map(get_polytope,get_reffes(trian))
  @notimplementedif !all(map(is_n_cube,ctype_polytope))
  cell_to_coords = get_cell_coordinates(trian)
  cell_to_bboxes = collect1d(lazy_map(a->(a[1],a[end]),cell_to_coords))
  cpp_f = @safe_cfunction(lsbufferφ, Float64, (ConstCxxRef{AlgoimUvector{Float64,2}},Float32))
  cpp_g = @safe_cfunction(lsbuffer∇φ, ConstCxxRef{AlgoimUvector{Float64,2}}, (ConstCxxRef{AlgoimUvector{Float64,2}},Float32))
  safecls = SafeCFunctionLevelSet{Int32(2)}(cpp_f,cpp_g)
  cell_to_quad = map(enumerate(cell_to_bboxes)) do (cell_id,bbox)
    bbmin, bbmax = bbox
    Quadrature(cell_id,bbmin,bbmax,safecls,args...;kwargs...)
  end
  CompressedArray(cell_to_quad,1:length(cell_to_quad))
end

function Quadrature(::Val{3},trian,::Algoim,args...;kwargs...)
  ctype_polytope = map(get_polytope,get_reffes(trian))
  @notimplementedif !all(map(is_n_cube,ctype_polytope))
  cell_to_coords = get_cell_coordinates(trian)
  cell_to_bboxes = collect1d(lazy_map(a->(a[1],a[end]),cell_to_coords))
  cpp_f = @safe_cfunction(lsbufferφ, Float64, (ConstCxxRef{AlgoimUvector{Float64,3}},Float32))
  cpp_g = @safe_cfunction(lsbuffer∇φ, ConstCxxRef{AlgoimUvector{Float64,3}}, (ConstCxxRef{AlgoimUvector{Float64,3}},Float32))
  safecls = SafeCFunctionLevelSet{Int32(3)}(cpp_f,cpp_g)
  cell_to_quad = map(enumerate(cell_to_bboxes)) do (cell_id,bbox)
    bbmin, bbmax = bbox
    Quadrature(cell_id,bbmin,bbmax,safecls,args...;kwargs...)
  end
  CompressedArray(cell_to_quad,1:length(cell_to_quad))
end

function Quadrature(cell_id::Int,
                    xmin::Point{N,T},
                    xmax::Point{N,T},
                    safecls::LevelSetFunction,
                    phi::LevelSetFunction,
                    degree::Int;
                    phase::Int=CUT) where {N,T}
  coords, weights = fill_quad_data(safecls,phi,xmin,xmax,phase,degree,cell_id)
  GenericQuadrature(coords,weights,"Algoim quadrature of degree $degree")
end

function is_cell_active(meas::Measure)
  has_non_empty_quad(x) = num_points(x) > 0
  lazy_map(has_non_empty_quad,get_data(meas.quad))
end

function restrict_measure(meas::Measure,trian::Triangulation)
  ocell_quad = meas.quad
  cell_quad   = lazy_map(Reindex(ocell_quad.cell_quad),trian.tface_to_mface)
  cell_point  = lazy_map(Reindex(ocell_quad.cell_point),trian.tface_to_mface)
  cell_weight = lazy_map(Reindex(ocell_quad.cell_weight),trian.tface_to_mface)
  dds = ocell_quad.data_domain_style
  ids = ocell_quad.integration_domain_style
  Measure(CellQuadrature(cell_quad,cell_point,cell_weight,trian,dds,ids))
end

function TriangulationAndMeasure(Ωbg::Triangulation,quad::Tuple)
  msg = "TriangulationAndMeasure can only receive the background triangulation"
  @notimplementedif num_cells(get_background_model(Ωbg)) != num_cells(Ωbg) msg
  dΩbg = Measure(Ωbg,quad,data_domain_style=PhysicalDomain())
  # RMK: This is a hack, but algoim interface does not let you 
  # know if a (given) cell intersects the interior of the level 
  # set. The hack consists in inferring this from the size of 
  # each quadrature. I do not expect this hack implies a lot of 
  # extra operations with regards to the proper way to do it.
  cell_to_is_active = is_cell_active(dΩbg)
  Ωᵃ = Triangulation(Ωbg,cell_to_is_active)
  dΩᵃ = restrict_measure(dΩbg,Ωᵃ)
  Ωᵃ,dΩᵃ,cell_to_is_active
end

function aggregate(bgtrian,cell_to_is_active,cell_to_is_cut,in_or_out;threshold=1.0)
  n_cells = length(cell_to_is_active)
  @assert n_cells == length(cell_to_is_cut)

  cell_to_unit_cut_meas = lazy_map(cell_to_is_active,cell_to_is_cut) do isa, isc
    !isa ? 0.0 : (isc ? 0.0 : 1.0)
  end

  cell_to_inoutcut = lazy_map(cell_to_is_active,cell_to_is_cut) do isa, isc
    !isa ? OUT : (isc ? CUT : IN)
  end

  cell_to_coords = get_cell_coordinates(bgtrian)
  model = get_background_model(bgtrian)
  topo = get_grid_topology(model)
  D = num_cell_dims(model)
  cell_to_faces = get_faces(topo,D,D-1)
  face_to_cells = get_faces(topo,D-1,D)
  # A hack follows to avoid constructing the actual facet_to_inoutcut array
  facet_to_inoutcut = fill(in_or_out,num_faces(model,D-1)) 

  _aggregate_by_threshold_barrier(
    threshold,cell_to_unit_cut_meas,facet_to_inoutcut,cell_to_inoutcut,
    in_or_out,cell_to_coords,cell_to_faces,face_to_cells)
end

function aggregate_narrow_band(bgtrian,cell_to_is_narrow,cell_to_active,cell_to_is_cut,in_or_out;threshold=1.0)
  n_cells = length(cell_to_is_narrow)
  @assert n_cells == length(cell_to_active)
  @assert n_cells == length(cell_to_is_cut)

  cell_to_unit_cut_meas = lazy_map(cell_to_active,cell_to_is_cut) do isa, isc
    ( isa & !isc ) ? 1.0 : 0.0
  end

  cell_to_inoutcut = lazy_map(cell_to_is_narrow,cell_to_active,cell_to_is_cut) do isn, isa, isc
    !isn ? OUT : ( ( isa & !isc ) ? IN : CUT )
  end

  cell_to_coords = get_cell_coordinates(bgtrian)
  model = get_background_model(bgtrian)
  topo = get_grid_topology(model)
  D = num_cell_dims(model)
  cell_to_faces = get_faces(topo,D,D-1)
  face_to_cells = get_faces(topo,D-1,D)
  # A hack follows to avoid constructing the actual facet_to_inoutcut array
  facet_to_inoutcut = fill(in_or_out,num_faces(model,D-1)) 

  _aggregate_by_threshold_barrier(
    threshold,cell_to_unit_cut_meas,facet_to_inoutcut,cell_to_inoutcut,
    in_or_out,cell_to_coords,cell_to_faces,face_to_cells)
end

using Gridap.Geometry: get_cell_to_parent_cell
using Gridap.CellData: get_cell_quadrature
using GridapEmbedded.AgFEM: compute_subcell_bbox
import GridapEmbedded.AgFEM: init_bboxes

function init_bboxes(cell_to_coords,cut_measure::Measure)
  bgcell_to_cbboxes = init_bboxes(cell_to_coords)
  quad = get_cell_quadrature(cut_measure)
  trian = get_triangulation(quad)
  model = get_active_model(trian)
  ccell_to_bgcell = get_cell_to_parent_cell(model)
  for (cc,bc) in enumerate(ccell_to_bgcell)
    bgcell_to_cbboxes[bc] = compute_subcell_bbox(quad.cell_point[cc])
  end
  bgcell_to_cbboxes
end

export init_bboxes

function compute_closest_point_projections(model::CartesianDiscreteModel,
                                           φ::AlgoimCallLevelSetFunction,
                                           num_refinements::Int;
                                           cppdegree::Int=2,limitstol::Float64=1.0e-8)
  ( num_refinements > 1 ) && begin
    model = refine(model,num_refinements)
    @error "Need to map nodes from lexicographic order to VEF order"
  end
  cdesc = get_cartesian_descriptor(model)
  partition = Int32[cdesc.partition...]
  xmin = cdesc.origin
  xmax = xmin + Point(cdesc.sizes .* partition)
  fill_cpp_data(φ,partition,xmin,xmax,cppdegree,limitstol)
end

function compute_closest_point_projections(Ω::Triangulation,φ,num_refinements;cppdegree::Int=2,limitstol::Float64=1.0e-8)
  compute_closest_point_projections(get_active_model(Ω),φ,num_refinements,cppdegree=cppdegree,limitstol=limitstol)
end

export fill_cpp_data
export fill_cpp_data_raw
export compute_closest_point_projections

function compute_normal_displacement(
    cps::AbstractVector{<:Point},
    phi::AlgoimCallLevelSetFunction,
    fun,
    dt::Float64,
    Ω::Triangulation)
  searchmethod = KDTreeSearch()
  cache1 = _point_to_cell_cache(searchmethod,Ω)
  x_to_cell(x) = _point_to_cell!(cache1, x)
  point_to_cell = lazy_map(x_to_cell, cps)
  cell_to_points, _ = make_inverse_table(point_to_cell, num_cells(Ω))
  cell_to_xs = lazy_map(Broadcasting(Reindex(cps)), cell_to_points)
  cell_point_xs = CellPoint(cell_to_xs, Ω, PhysicalDomain())
  fun_xs = evaluate(fun,cell_point_xs)
  nΓ_xs = evaluate(normal(phi,Ω),cell_point_xs)
  cell_point_disp = lazy_map(Broadcasting(⋅),fun_xs,nΓ_xs)
  cache_vals = array_cache(cell_point_disp)
  cache_ctop = array_cache(cell_to_points)
  disps = zeros(Float64,length(cps))
  for cell in 1:length(cell_to_points)
    pts = getindex!(cache_ctop,cell_to_points,cell)
    vals = getindex!(cache_vals,cell_point_disp,cell)
    for (i,pt) in enumerate(pts)
      val = vals[i]
      disps[pt] = dt * val
    end
  end
  disps
end

export compute_normal_displacement

end # module