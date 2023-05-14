function init_bboxes(cell_to_coords,args...;kwargs...)
  @abstractmethod
end

function init_bboxes(cell_to_coords)
  # RMK: Assuming first node is min and last node is max of BBox
  map(x-> [ x[1], x[end] ], vec(cell_to_coords) )
end

function init_bboxes(cell_to_coords,cut::EmbeddedDiscretization;in_or_out=IN)
  bgcell_to_cbboxes = init_bboxes(cell_to_coords)
  cut_bgtrian = Triangulation(cut,CUT,cut.geo)
  cut_bgmodel = get_active_model(cut_bgtrian)
  ccell_to_bgcell = get_cell_to_parent_cell(cut_bgmodel)
  ccell_to_cbboxes = init_cut_bboxes(cut,ccell_to_bgcell,in_or_out)
  for (cc,cbb) in enumerate(ccell_to_cbboxes)
    bgcell_to_cbboxes[ccell_to_bgcell[cc]] = cbb
  end
  bgcell_to_cbboxes
end

function init_cut_bboxes(cut,ccell_to_bgcell,in_or_out)
  subcell_to_inout   = compute_subcell_to_inout(cut,cut.geo)
  subcell_is_in      = lazy_map(i->i==in_or_out,subcell_to_inout)
  subcell_to_bgcell  = cut.subcells.cell_to_bgcell
  inscell_to_subcell = findall(subcell_is_in)
  inscell_to_bgcell  = lazy_map(Reindex(subcell_to_bgcell),inscell_to_subcell)
  inscell_to_bboxes  = init_subcell_bboxes(cut,inscell_to_subcell)
  lazy_map(ccell_to_bgcell) do bg
    bg_to_scs = findall(map(x->x==bg,inscell_to_bgcell))
    bg_to_bbs = inscell_to_bboxes[bg_to_scs]
    compute_bgcell_cut_bbox(bg_to_bbs)
  end
end

function init_subcell_bboxes(cut,inscell_to_subcell)
  subcell_to_points = cut.subcells.cell_to_points
  point_to_coords   = cut.subcells.point_to_coords
  inscell_to_points = lazy_map(Reindex(subcell_to_points),inscell_to_subcell)
  inscell_to_coords = lazy_map( #
    Broadcasting(Reindex(point_to_coords)),inscell_to_points)
  lazy_map(compute_subcell_bbox,inscell_to_coords)
end

function compute_subcell_bbox(subcell_to_coords)
  c(i) = map(x -> x[i],subcell_to_coords)
  mins = [minimum(c(i)) for i in 1:length(first(subcell_to_coords))]
  maxs = [maximum(c(i)) for i in 1:length(first(subcell_to_coords))]
  [VectorValue(mins),VectorValue(maxs)]
end

function compute_bgcell_cut_bbox(bgcell_to_bboxes)
  c = map(x -> x[1].data,bgcell_to_bboxes)
  mins = min.(c...)
  c = map(x -> x[2].data,bgcell_to_bboxes)
  maxs = max.(c...)
  [VectorValue(mins),VectorValue(maxs)]
end

function compute_bboxes!(root_to_agg_bbox,cell_to_root)
  cell_to_bbox = root_to_agg_bbox
  for (c,r) in enumerate(cell_to_root)
    ( ( r == 0 ) | ( c == r ) ) && continue
    bbmin = min.(root_to_agg_bbox[r][1].data,cell_to_bbox[c][1].data)
    bbmax = max.(root_to_agg_bbox[r][2].data,cell_to_bbox[c][2].data)
    root_to_agg_bbox[r] = [bbmin,bbmax]
  end
end

function reset_bboxes_at_cut_cells!(root_to_agg_bbox,
                                    cell_to_root,
                                    cell_to_coords)
  for (c,r) in enumerate(cell_to_root)
    ( ( r == 0 ) | ( c == r ) ) && continue
    root_to_agg_bbox[c] = [cell_to_coords[c][1],cell_to_coords[c][end]]
  end
end

function compute_cell_bboxes(model,cell_to_root,args...;kwargs...)
  @abstractmethod
end

function compute_cell_bboxes(model::DiscreteModelPortion,
                             cell_to_root,args...;kwargs...)
  compute_cell_bboxes(get_parent_model(model),cell_to_root,args...;kwargs...)
end

function compute_cell_bboxes(model::DiscreteModel,
                             cell_to_root,args...;kwargs...)
  trian = Triangulation(model)
  compute_cell_bboxes(trian,cell_to_root,args...;kwargs...)
end

function compute_cell_bboxes(trian::Triangulation,
                             cell_to_root,args...;kwargs...)
  cell_to_coords = get_cell_coordinates(trian)
  root_to_agg_bbox = init_bboxes(cell_to_coords,args...;kwargs...)
  compute_bboxes!(root_to_agg_bbox,cell_to_root)
  reset_bboxes_at_cut_cells!(root_to_agg_bbox,cell_to_root,cell_to_coords)
  root_to_agg_bbox
end

function compute_bbox_dfaces(model::DiscreteModel,cell_to_agg_bbox)
  gt = get_grid_topology(model)
  bboxes = Array{eltype(cell_to_agg_bbox),1}[]
  D = num_dims(gt)
  for d = 1:D-1
    dface_to_Dfaces = get_faces(gt,d,D)
    d_bboxes = map(1:num_faces(gt,d)) do face
      _compute_bbox_dface(dface_to_Dfaces,cell_to_agg_bbox,face)
    end
    bboxes = push!(bboxes,d_bboxes)
  end
  bboxes = push!(bboxes,cell_to_agg_bbox)
end

function _compute_bbox_dface(dface_to_Dfaces,cell_to_agg_bbox,i)
  cells_around_dface_i = getindex(dface_to_Dfaces,i)
  bboxes_around_dface_i = cell_to_agg_bbox[cells_around_dface_i]
  bbmins = map( i->i[1].data, bboxes_around_dface_i )
  bbmaxs = map( i->i[2].data, bboxes_around_dface_i )
  [min.(bbmins...),max.(bbmaxs...)]
end

function _compute_cell_to_dface_bboxes(model::DiscreteModel,dbboxes)
  gt = get_grid_topology(model)
  trian = Triangulation(model)
  ctc = get_cell_coordinates(trian)
  bboxes = map(1:num_cells(model)) do cell
    __compute_cell_to_dface_bboxes(gt,ctc,dbboxes,cell)
  end
  CellPoint(bboxes,trian,PhysicalDomain()).cell_ref_point
end

function __compute_cell_to_dface_bboxes(gt::GridTopology,ctc,dbboxes,cell::Int)
  cdbboxes = eltype(eltype(eltype(dbboxes)))[]
  D = num_dims(gt)
  Dface_to_0faces = get_faces(gt,D,0)
  for face in getindex(Dface_to_0faces,cell)
    cdbboxes = vcat(cdbboxes,[ctc[cell][1],ctc[cell][end]])
  end
  for d = 1:D-1
    Dface_to_dfaces = get_faces(gt,D,d)
    for face in getindex(Dface_to_dfaces,cell)
      cdbboxes = vcat(cdbboxes,dbboxes[d][face])
    end
  end
  cdbboxes = vcat(cdbboxes,[dbboxes[D][cell][1],dbboxes[D][cell][end]])
end

function compute_cell_to_dface_bboxes(model,cell_to_root,args...;kwargs...)
  @abstractmethod
end

function compute_cell_to_dface_bboxes(model::DiscreteModel,
                                      cell_to_root,args...;kwargs...)
  cbboxes = compute_cell_bboxes(model,cell_to_root,args...;kwargs...)
  dbboxes = compute_bbox_dfaces(model,cbboxes)
  _compute_cell_to_dface_bboxes(model,dbboxes)
end

function compute_cell_to_dface_bboxes(model::DiscreteModelPortion,
                                      cell_to_root,args...;kwargs...)
  pmodel = get_parent_model(model)
  bboxes = compute_cell_to_dface_bboxes(pmodel,cell_to_root,args...;kwargs...)
  bboxes[get_cell_to_parent_cell(model)]
end