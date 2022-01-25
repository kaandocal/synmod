using Catalyst
using DifferentialEquations
using AdvancedMH, MCMCChains
using LinearAlgebra

include("nbmixture.jl")
include("synmod.jl")

include("ts/ts.jl")

##

samplesize = 10000
ncomps = 4

data_sim = zeros(Int, length(tt), 2, samplesize)

function logl_synmod(params; mixture_kwargs=Dict(:maxiter=>1000),
                             hmm_kwargs=Dict(:maxiter=>10), kwargs...)
    simulate!(data_sim, params)
    syn = SyntheticModelPS(tt, [ncomps, ncomps])
   
    fit!(syn, data_sim; maxiter=1000)

    loglikelihood(syn, data_obs)
end

##

insupport(ps) = all(ranges[:,1] .<= ps .<= ranges[:,2])

pdensity(ps) = insupport(ps) != 0 ? logl_synmod(ps) : -Inf

model = DensityModel(pdensity)

D = size(ranges, 1)
trker_std = 0.01 .* (ranges[:,2] .- ranges[:,1])
spl = RWMH(MvNormal(zeros(D), diagm(trker_std .^ 2)))

##

init_params = rand(prior)
@time chain = sample(model, spl, 100000; init_params=init_params, chain_type=Chains)
CSV.write("data/ts/posterior_sm.csv", Tables.table(chain.value.data[:,:,1]))
