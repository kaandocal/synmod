using Catalyst
using DifferentialEquations
using AdvancedMH, MCMCChains
using LinearAlgebra

include("nbmixture.jl")
include("synmod.jl")

experiment = "nfl"
include("afl/afl.jl")

##

samplesize = 5000
ncomps = 5

data_sim = zeros(Int, length(tt), 1, samplesize)

extract_margs(x) = x[2]

function logl_synmod_ts(params; mixture_kwargs = Dict(:maxiter=>750),
                                hmm_kwargs = Dict(:maxiter=>10), kwargs...)
    simulate!(extract_margs, data_sim, params)

    syn = SyntheticModelTS(tt, [ncomps])
    
    fit!(syn, data_sim; mixture_kwargs, hmm_kwargs, kwargs...)
    
    loglikelihood(syn, reshape(data_obs, (size(data_obs,1), 1, size(data_obs,2))))
end

##

insupport(ps) = all(ranges[:,1] .<= ps .<= ranges[:,2])
pdensity(ps) = pdf(prior, ps) != 0 ? logl_synmod_ts(ps) + logpdf(prior, ps) : -Inf

model = DensityModel(pdensity)

D = size(ranges, 1)
trker_std = 0.05 .* sqrt.(diag(cov(prior)))
trker_std[4] /= 5
spl = RWMH(MvNormal(zeros(D), diagm(trker_std .^ 2)))

##

init_params = rand(prior)
@time chain = sample(model, spl, 50000; init_params=init_params, chain_type=Chains)
CSV.write("data/afl/posterior_$(experiment)_sm.csv", Tables.table(chain.value.data[:,:,1]))
