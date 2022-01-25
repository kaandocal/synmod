using Catalyst
using DifferentialEquations
using AdvancedMH, MCMCChains
using LinearAlgebra
using GpABC, Distances

include("nbmixture.jl")
include("synmod.jl")

include("ts/ts.jl")

##

data_sim = similar(data_obs)

extract_ss(data) = [ mean(data, dims=3)..., 
                     std(data, dims=3)...,
                     mean(data[:,1,:] .* data[:,2,:], dims=2)... ]

reference_data = extract_ss(data_obs)

function simulator_function(params; kwargs...)
    simulate!(data_sim, params)

    extract_ss(data_sim)'
end

##

threshold_schedule = [ 20.0, 15., 10., 7.5, 5.0 ]

distance_function = (x,y) -> weuclidean(x, y, 1 ./ reference_data)

sim_abc_res = SimulatedABCSMC(reference_data', simulator_function,
                              prior.v, threshold_schedule, 5000;
                              max_iter = Int(1e8),
                              distance_function = distance_function,
                              progress_every=10000)

chain_data = vcat(hcat.(sim_abc_res.population, sim_abc_res.weights)...)
CSV.write("data/ts/posterior_abc.csv", Tables.table(chain_data))
