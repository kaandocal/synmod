using Catalyst
using DifferentialEquations
using MomentClosure
using AdvancedMH, MCMCChains
using LinearAlgebra
using PyPlot

using Sundials

include("nbmixture.jl")
include("synmod.jl")

include("ts/ts.jl")
include("mbi.jl")

##

obs_moments = hcat(mean(data_obs[:,1,:], dims=2)[:],
                   mean(data_obs[:,2,:], dims=2)[:],
                   mean(data_obs[:,1,:] .^ 2, dims=2)[:],
                   mean(data_obs[:,2,:] .^ 2, dims=2)[:],
                   mean(data_obs[:,1,:] .* data_obs[:,2,:], dims=2)[:])

##


rn_lma = @reaction_network begin
    σ_b1, Gu_A --> Gb_A
    σ_u1, Gb_A --> Gu_A + Pn_B
    ρ_m1, Gu_A --> Gu_A + Mn_A
    12, Mn_A --> Mc_A
    3, Mc_A --> 0
    ρ_p1, Mc_A --> Mc_A + Pc_A
    (4, 4), Pc_A <--> Pn_A
    2, Pc_A --> 0

    σ_b2, Gu_B --> Gb_B
    σ_u2, Gb_B --> Gu_B + Pn_A
    ρ_m2, Gu_B --> Gu_B + Mn_B
    16, Mn_B --> Mc_B
    1, Mc_B --> 0
    ρ_p2, Mc_B --> Mc_B + Pc_B
    (4, 4), Pc_B <--> Pn_B
    3, Pc_B --> 0
end σ_b1 σ_u1 ρ_m1 ρ_p1 σ_b2 σ_u2 ρ_m2 ρ_p2

u0_compact = [ u0...]


##

raw_eqs = generate_raw_moment_eqs(rn, 4)
bern_eqs = bernoulli_moment_eqs(raw_eqs, [1,3,8,9])

mom_eqs = moment_closure(raw_eqs, "normal", [1,3,8,9])

u0map = deterministic_IC(u0_compact, mom_eqs)
prob = ODEProblem(mom_eqs, u0map, (0.0, last(tt)), gt_params)

##

LMA_eqs, eff_params = linear_mapping_approximation(rn, rn_lma, [1, 3, 8, 9], 4)

u0map = deterministic_IC(u0_compact, LMA_eqs)

prob = ODEProblem(LMA_eqs, u0map, (0.0, last(tt)), gt_params)

##

momentlist = [ 0 0 0 0 0 1 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 1
               0 0 0 0 0 2 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 2
               0 0 0 0 0 1 0 0 0 0 0 1 ]

function logl_moments(params, solver=CVODE_BDF(linear_solver=:GMRES))
    _prob = remake(prob,p=params)
    sol = solve(_prob, solver, saveat=tt)
    
    try
        dists = getmomentdist(sol, rn, LMA_eqs, momentlist, ones(length(tt)) .* size(data_obs,3))
        ret = sum(logpdf(dists[i], obs_moments[i,:]) for i in 1:length(tt))
        return ret
    catch PosDefException
        @warn "Matrix not positive definite"
        return -Inf
    end
end

##

insupport(ps) = all(ranges[:,1] .<= ps .<= ranges[:,2])

pdensity(ps) = insupport(ps) ? logl_moments(ps) : -Inf

model = DensityModel(pdensity)

D = size(ranges, 1)
trker_std = 0.001 .* (ranges[:,2] .- ranges[:,1])
spl = RWMH(MvNormal(zeros(D), diagm(trker_std .^ 2)))

##

init_params = rand(prior)
@time chain = sample(model, spl, 1000000; init_params=init_params, chain_type=Chains)
CSV.write("data/ts/posterior_mbi.csv", Tables.table(chain.value.data[:,:,1]))
