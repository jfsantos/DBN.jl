include(joinpath(dirname(@__FILE__), "../src/DBN.jl")
using HDF5, JLD

data_layer = HDF5DataLayer(name="train-data", source="train.txt", batch_size=1024)
source = HDF5DataSource(data_layer)
dbn = DBN([64*11, 2048, 2048, 2048])

fit(dbn, source)

file = jldopen("DBN3.jld", "w")
@write file dbn
close(file)

