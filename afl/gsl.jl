using Catalyst
using DifferentialEquations
using AdvancedMH, MCMCChains
using LinearAlgebra

include("nbmixture.jl")
include("synmod.jl")

experiment = "nfl"
include("afl/afl.jl")

##

samplesize = 3000

data_sim = zeros(Int, length(tt), 1, samplesize)

extract_margs(x) = x[2]

function logl_gsl(params; kwargs...)
    simulate!(extract_margs, data_sim, params)

    data_flat = reshape(data_sim, (length(tt) * 2, size(data_sim, 3)))

    means = mean(data_flat, dims=2)[:]
    cov_matrix = cov(data_flat')

    dist = MvNormal(means, cov_matrix)

    sum(logpdf(dist, data_obs))
end

##

insupport(ps) = all(ranges[:,1] .<= ps .<= ranges[:,2])
pdensity(ps) = pdf(prior, ps) != 0 ? logl_gsl(ps) + logpdf(prior, ps) : -Inf

model = DensityModel(pdensity)

D = size(ranges, 1)
trker_std = 0.1 .* sqrt.(diag(cov(prior)))
spl = RWMH(MvNormal(zeros(D), diagm(trker_std .^ 2)))

##

init_params = rand(prior)
@time chain = sample(model, spl, 50000; init_params=init_params, chain_type=Chains)
CSV.write("data/afl/posterior_$(experiment)_gsl.csv", Tables.table(chain.value.data[:,:,1]))
