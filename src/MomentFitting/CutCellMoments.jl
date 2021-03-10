struct CutCellMoments
  data::Vector{Vector{Float64}}
  bgcell_to_cut_cell::Vector{Int32}
end

function CutCellMoments(model::RestrictedDiscreteModel,
                        facet_moments::DomainContribution)
  fi = [ testitem(array) for (trian,array) in facet_moments.dict ]
  li = map(length,fi)
  @assert all(li .== first(li))
  cell_to_parent_cell = get_cell_to_parent_cell(model)
  data = [ zero(first(fi)) for i in 1:length(cell_to_parent_cell) ]
  bgcell_to_cut_cell = zeros(Int32,num_cells(get_parent_model(model)))
  bgcell_to_cut_cell[cell_to_parent_cell] .= 1:length(cell_to_parent_cell)
  CutCellMoments(data,bgcell_to_cut_cell)
end

function compute_cell_moments(cut::EmbeddedDiscretization{D,T},
                              degree::Int) where{D,T}
  bgtrian = Triangulation(cut.bgmodel)
  b = MonomialBasis{D}(T,degree)
  v = get_monomial_cell_field(b,bgtrian)
  cut_bgmodel = DiscreteModel(cut,cut.geo,CUT)
  J = compute_monomial_domain_contribution(cut,degree,v)
  in = compute_monomial_cut_cell_moments(cut_bgmodel,J,b)
  p = check_and_get_polytope(cut)
  change = compute_change_of_basis(b,p,degree)
  C = Fill(change,length(in))
  lazy_map(*,C,in)
end

function compute_monomial_domain_contribution(cut::EmbeddedDiscretization,
                                              degree::Int,
                                              v::CellField)
  Γᵉ = EmbeddedBoundary(cut)
  Λ  = GhostSkeleton(cut)
  cutf = cut_facets(cut.bgmodel,cut.geo)
  Γᶠ = SkeletonTriangulation(cutf,Λ,cut.geo,CUTIN)
  Γᵇ = SkeletonTriangulation(cutf,Λ,cut.geo,IN)
  bgfacet_to_inoutcut = compute_bgfacet_to_inoutcut(cutf,cut.geo)
  bgfacet_to_mask = lazy_map( a -> a == IN, bgfacet_to_inoutcut )
  Γᵒ = BoundaryTriangulation(cut.bgmodel,bgfacet_to_mask)

  dΓᵉ = Measure(Γᵉ,degree)
  dΓᶠ = SkeletonPair(Measure(Γᶠ.⁺,degree),Measure(Γᶠ.⁻,degree))
  dΓᵇ = SkeletonPair(Measure(Γᵇ.⁺,degree),Measure(Γᵇ.⁻,degree))
  dΓᵒ = Measure(Γᵒ,degree)

  cᵉ = compute_hyperplane_coeffs(Γᵉ)
  cᶠ = compute_hyperplane_coeffs(Γᶠ)
  cᵇ = compute_hyperplane_coeffs(Γᵇ)
  cᵒ = compute_hyperplane_coeffs(Γᵒ)

  @check num_cells(Γᵉ) > 0
  J = ∫(cᵉ*v)*dΓᵉ +
      ∫(cᶠ.⁺*v)*dΓᶠ.⁺ + ∫(cᶠ.⁻*v)*dΓᶠ.⁻
  if num_cells(Γᵇ) > 0
    J += ∫(cᵇ.⁺*v)*dΓᵇ.⁺ + ∫(cᵇ.⁻*v)*dΓᵇ.⁻
  end
  if num_cells(Γᵒ) > 0
    J += ∫(cᵒ*v)*dΓᵒ
  end
  J

end

function compute_monomial_cut_cell_moments(model::RestrictedDiscreteModel,
                                           facet_moments::DomainContribution,
                                           b::MonomialBasis{D,T}) where {D,T}
  cut_cell_to_moments = CutCellMoments(model,facet_moments)
  for (trian,array) in facet_moments.dict
    add_facet_moments!(cut_cell_to_moments,trian,array)
  end
  o = get_terms_degrees(b)
  q = 1 ./ ( D .+ o )
  [ q .* d for d in cut_cell_to_moments.data ]
end

function add_facet_moments!(ccm::CutCellMoments,trian,array::AbstractArray)
  @abstractmethod
end

function add_facet_moments!(ccm::CutCellMoments,
                            trian::SubFacetTriangulation,
                            array::AbstractArray)
  add_facet_moments!(ccm,trian.subfacets,array)
end

function add_facet_moments!(ccm::CutCellMoments,
                            sfd::SubFacetData,
                            array::AbstractArray)
  facet_to_cut_cell = lazy_map(Reindex(ccm.bgcell_to_cut_cell),sfd.facet_to_bgcell)
  for i = 1:length(facet_to_cut_cell)
    ccm.data[facet_to_cut_cell[i]] += array[i]
  end
end

function add_facet_moments!(ccm::CutCellMoments,
                            trian::SubFacetBoundaryTriangulation,
                            array::AbstractArray)
  add_facet_moments!(ccm,trian.facets,array)
end

function add_facet_moments!(ccm::CutCellMoments,
                            trian::BoundaryTriangulation,
                            array::AbstractArray)
  add_facet_moments!(ccm,trian.glue,array)
end

function add_facet_moments!(ccm::CutCellMoments,
                            glue::FaceToCellGlue,
                            array::AbstractArray)
  facet_to_cut_cell = lazy_map(Reindex(ccm.bgcell_to_cut_cell),glue.face_to_cell)
  cell_to_is_cut = findall(lazy_map(i->(i>0),facet_to_cut_cell))
  facet_to_cut_cell = lazy_map(Reindex(facet_to_cut_cell),cell_to_is_cut)
  l = length(facet_to_cut_cell)
  for i = 1:l
    ccm.data[facet_to_cut_cell[i]] += array[cell_to_is_cut[i]]
  end
end

function compute_change_of_basis(b::MonomialBasis{D,T},
                                 p::Polytope,
                                 degree::Int) where {D,T}
  orders = tfill(degree,Val{D}())
  nodes, _ = compute_nodes(p,orders)
  dofs = LagrangianDofBasis(T,nodes)
  transpose((inv(evaluate(dofs,b))))
end

@inline function check_and_get_polytope(cut::EmbeddedDiscretization)
  _check_and_get_polytope(cut.bgmodel.grid)
end

function get_monomial_cell_field(b::MonomialBasis{D,T},
                                 trian::Triangulation) where {D,T}
  i = Matrix{eltype(T)}(I,length(b),length(b))
  l = linear_combination(i,b)
  m = Fill(l,num_cells(trian))
  v = GenericCellField(m,trian,ReferenceDomain())
end

@inline function get_terms_degrees(b::MonomialBasis)
  [ _get_terms_degrees(c) for c in b.terms ]
end

function _get_terms_degrees(c::CartesianIndex)
  d = 0
  for i in 1:length(c)
    d += (c[i]-1)
  end
  d
end
