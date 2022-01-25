using Catalyst
using DifferentialEquations
using AdvancedMH, MCMCChains
using LinearAlgebra
using GpABC, Distances

include("nbmixture.jl")
include("synmod.jl")

experiment = "nfl"
include("afl/afl.jl")

##

samplesize = size(data_obs, 2)

data_sim = zeros(Int, length(tt), 1, samplesize)

extract_ss(data) = [ mean(data, dims=(2,3))..., std(data, dims=(2,3))[:]... ]

reference_data = extract_ss(reshape(data_obs, (4, 1, size(data_obs, 2))))

extract_margs(x) = x[2]

function simulator_function(params; kwargs...)
    simulate!(extract_margs, data_sim, params)

    extract_ss(data_sim)'
end

##

threshold_schedule = [ 1.5, 1.25, 1.0, 0.75 ]

distance_function = (x,y) -> weuclidean(x, y, 1 ./ reference_data)

sim_abc_res = SimulatedABCSMC(reference_data', simulator_function,
                              prior.v, threshold_schedule, 1000;
                              max_iter = Int(1e8),
                              distance_function = distance_function,
                              progress_every=10000)

chain_data = vcat(hcat.(sim_abc_res.population, sim_abc_res.weights)...)
CSV.write("data/afl/posterior_$(experiment)_abc.csv", Tables.table(chain_data2))
