using Catalyst
using DifferentialEquations
using MomentClosure
using AdvancedMH, MCMCChains
using LinearAlgebra
using Sundials


include("nbmixture.jl")
include("synmod.jl")

include("mbi.jl")

experiment = "nfl"
include("afl/afl.jl")

##

obs_moments = hcat(mean(data_obs, dims=2),
                   mean(data_obs .^ 2, dims=2))

if !(experiment in ["nfl"])
    rn_lma = @reaction_network begin
        σ_u * (1 - G), 0 --> G + P
        σ_b, G --> 0
        ρ_u, G --> G + P
        ρ_b * (1 - G), 0 --> P
        1, P --> 0
    end σ_u σ_b ρ_u ρ_b     
else
    rn_lma = @reaction_network begin
        σ_u * (1 - G), 0 --> G + P
        σ_b, G --> 0
        ρ_u * b / (1+b)^2, G --> G + P
        ρ_b * b / (1+b)^2 * (1 - G), 0 --> P
        1, P --> 0
    end σ_u σ_b ρ_u ρ_b b

    add_bursty_reactions!(rn_lma, ρ_u, ρ_b, b)
end

LMA_eqs, eff_params = linear_mapping_approximation(rn, rn_lma, [1], 4);

u0map = deterministic_IC([ u0... ], LMA_eqs)

prob = ODEProblem(LMA_eqs, u0map, (0.0, last(tt)), gt_params)

##

momentlist = [ 0 1
               0 2 ]

function logl_moments(params)
    _prob = remake(prob,p=params)
    sol = solve(_prob, KenCarp4(), saveat=tt)
    
    dists = getmomentdist(sol, rn, LMA_eqs, momentlist, 
                          ones(size(data_obs,1)) .* size(data_obs, 2))

    ret = 0
    for i in 1:length(tt)
        ret += logpdf(dists[i], obs_moments[i,:])
    end

    ret
end

##

insupport(ps) = all(ranges[:,1] .<= ps .<= ranges[:,2])
pdensity(ps) = pdf(prior, ps) != 0 ? logl_moments(ps) + logpdf(prior, ps) : -Inf

model = DensityModel(pdensity)

D = size(ranges, 1)
trker_std = 0.15 .* sqrt.(diag(cov(prior)))
spl = RWMH(MvNormal(zeros(D), diagm(trker_std .^ 2)))

##

init_params = rand(prior)
chain = sample(model, spl, 50000; init_params=init_params, chain_type=Chains)
CSV.write("data/afl/posterior_$(experiment)_mbi.csv", Tables.table(chain.value.data[:,:,1]))
