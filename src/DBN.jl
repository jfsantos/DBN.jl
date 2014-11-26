using Mocha, Boltzmann

type HDF5DataSource
  sys::System
  data_layer::HDF5DataLayer
  state::Mocha.HDF5DataLayerState

  HDF5DataSource(layer::HDF5DataLayer) = begin
    sys = System(CPUBackend())
    inputs = Array(Blob, 0)
    diffs = Array(Blob, 0)
    state = setup(sys, layer, inputs, diffs)
    new(sys, layer, state)
  end
end

function get_data_batch(source::HDF5DataSource)
  forward(source.sys, source.state, Array(Blob, 0))
  X = source.state.blobs[1].data
  reshape(X, (size(X,1)*size(X,2), size(X,4)))
end

# TODO: data source has to always output data between 0 and 1! seems to be the case for our data.
function fit(rbm::Boltzmann.RBM, source::HDF5DataSource, lr=0.1, n_iter=10, n_gibbs=1)
  batch_size = source.data_layer.batch_size
  n_samples = size(source.state.curr_hdf5_file["data"],4)
  n_batches = int(ceil(n_samples / batch_size))
  w_buf = zeros(size(rbm.W))
  for itr=1:n_iter
    for i=1:n_batches
      info("Iteration $(itr): fitting batch $(i)/$(n_batches)")
      batch = convert(Array{Float64,2}, get_data_batch(source))
      Boltzmann.fit_batch!(rbm, batch, buf=w_buf, n_gibbs=n_gibbs)
    end
    #@printf("Iteration #%s, pseudo-likelihood = %s\n",
    #        itr, mean(Boltzmann.score_samples(rbm, X)))
   end
end

type DBN
  rbms::Vector{GRBM}

  DBN(dims::Vector{Int}) = begin
    rbms = Array(GRBM, 0)
    for k=1:length(dims)-1
      push!(rbms, GRBM(dims[k],dims[k+1]))
    end
    new(rbms)
  end
end

mh(rbm::GRBM, vis::Array{Float64, 2}) = Boltzmann.mean_hiddens(rbm, vis)

function mh_at_layer(dbn::DBN, batch::Array{Float64, 2}, layer::Int)
  hiddens = Array(Array{Float64, 2}, layer)
  hiddens[1] = mh(dbn.rbms[1], batch)
  for k=2:layer
    hiddens[k] = mh(dbn.rbms[k], hiddens[k-1])
  end
  hiddens[end]
end

function fit(dbn::DBN, source::HDF5DataSource, lr=0.1, n_iter=10, n_gibbs=1)
  batch_size = source.data_layer.batch_size
  n_samples = size(source.state.curr_hdf5_file["data"],4)
  n_batches = int(ceil(n_samples / batch_size))
  for k = 1:length(dbn.rbms)
    println("Training layer $k/$(length(dbn.rbms))")
    w_buf = zeros(size(dbn.rbms[k].W))
    for itr=1:n_iter
      for i=1:n_batches
        info("Iteration $(itr): fitting batch $(i)/$(n_batches)")
        batch = convert(Array{Float64,2}, get_data_batch(source))
        if k == 1
          input = batch
        else
          input = mh_at_layer(dbn, batch, k)
        end
        Boltzmann.fit_batch!(dbn.rbms[k], input, buf=w_buf, n_gibbs=n_gibbs)
      end
    end
  end
end
