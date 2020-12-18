module AggregateBoundingBoxesTests

using Gridap
using GridapEmbedded
using GridapEmbedded.AgFEM

const R = 0.4
geom = disk(R,x0=Point(0.5,0.5))
n = 5
partition = (n,n)

domain = (0,1,0,1)
bgmodel = CartesianDiscreteModel(domain,partition)

cutdisc = cut(bgmodel,geom)

strategy = AggregateAllCutCells()

aggregates = aggregate(strategy,cutdisc,geom)

colors = color_aggregates(aggregates,bgmodel)

trian = Triangulation(bgmodel)

coords = collect(get_cell_coordinates(trian))

bboxes = compute_aggregate_bboxes(trian,aggregates)

writevtk(trian,"trian",
         celldata=["cellin"=>aggregates,"color"=>colors])

end # module
