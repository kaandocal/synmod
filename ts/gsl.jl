using Catalyst
using DifferentialEquations
using AdvancedMH, MCMCChains
using LinearAlgebra

include("nbmixture.jl")
include("synmod.jl")

include("ts/ts.jl")

##

samplesize = 10000

data_sim = zeros(Int, length(tt), 2, samplesize)

function logl_gsl(params; kwargs...)
    simulate!(data_sim, params)

    data_flat = reshape(data_sim, (length(tt) * 2, size(data_sim, 3)))

    means = mean(data_flat, dims=2)[:]
    cov_matrix = cov(data_flat')

    dist = MvNormal(means, cov_matrix)

    sum(logpdf(dist, reshape(data_obs, (length(tt) * 2, size(data_obs, 3)))))
end

##

insupport(ps) = all(ranges[:,1] .<= ps .<= ranges[:,2])

pdensity(ps) = insupport(ps) ? logl_gsl(ps) : -Inf

model = DensityModel(pdensity)

D = size(ranges, 1)
trker_std = 0.01 .* (ranges[:,2] .- ranges[:,1])
spl = RWMH(MvNormal(zeros(D), diagm(trker_std .^ 2)))

##

init_params = rand(prior)
@time chain = sample(model, spl, 100000; init_params=init_params, chain_type=Chains)
CSV.write("data/ts/posterior_gsl.csv", Tables.table(chain.value.data[:,:,1]))
