using Catalyst
using DifferentialEquations, DiffEqBase
using MomentClosure
using BlackBoxOptim
using AdvancedMH, MCMCChains
using LinearAlgebra
using PyPlot
using FileIO, JLD2

include("nbmixture.jl")
include("synmod.jl")

include("hog1p/utils.jl")

##

data_obs = hog1pload("data/Data_Gregor_04_27_2012.mat")
data_obs = Array{Float64}(data_obs)

ncells = [ sum(data_obs[i,1,:]) for i in 1:size(data_obs,1) ]

##

gt_params = [ 1.3, 0.0067, 0.13, 
              3200, -3200*2.4, 0.027, 0.038,
              0.00078, 0.012, 0.99, 0.054,
              0.0049]

function convert_params(ps)
    ret = copy(ps)
    ret[1:4] = 10. .^ ret[1:4]
    ret[6:end] = 10. .^ ret[6:end]
    ret
end

Nmax = size(data_obs, 3) - 1

tt = Float64[ 60, 120, 240, 360, 480, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300 ]

tmax = tt[end]

if jacinfo != nothing && jacinfo["dims"] != (Nmax+1,4)
    jacinfo = nothing
end

m1s = zeros(length(tt))
m2s = zeros(length(tt))

xx = 0:Nmax
for i in 1:length(tt)
    m1s[i] = sum(data_obs[i,1,:] .* xx) / sum(data_obs[i,1,:])
    m2s[i] = sum(data_obs[i,1,:] .* (xx .^ 2)) / sum(data_obs[i,1,:])
end

obs_moments = [ m1s m2s ]

##

@parameters k12 k23 k34 
@parameters a21 b21 k32 k43 
@parameters r1 r2 r3 r4 delta

rs = @reaction_network begin
    (k12, max(0, a21 + b21 * hog1p(t))), G1 <--> G2
    (k23, k32), G2 <--> G3
    (k34, k43), G3 <--> G4
    r1, G1 --> G1 + P
    r2, G2 --> G2 + P
    r3, G3 --> G3 + P
    r4, G4 --> G4 + P
    delta, P --> 0
end k12 k23 k34 a21 b21 k32 k43 r1 r2 r3 r4 delta

##

momeqs = generate_raw_moment_eqs(rs, 4)

#

u0 = [1,0,0,0,0]
u0map = deterministic_IC(u0, momeqs)

prob = ODEProblem(momeqs, u0map, (0.0, tmax), gt_params)

##

function getmomentdist(sol::DESolution, rs, momeqs::MomentClosure.MomentEquations, 
                       obs_moments::AbstractMatrix{Int}, ncells::AbstractVector)
    @argcheck size(obs_moments, 2) == length(species(rs))
    ret = Array{MultivariateDistribution}(undef, length(sol.t))
    for (i, t) in enumerate(sol.t)
        ret[i] = getmomentdist(sol.u[i], rs, momeqs, obs_moments, ncells[i])
    end
    ret
end

function extractmoment(raw_moments::AbstractVector, moment::Tuple, 
                       momeqs::MomentClosure.MomentEquations)
    mu = momeqs.μ[moment]
    idx = findfirst(x -> isequal(x, mu), states(momeqs.odes))
    idx !== nothing || error("Moment $(moment) not found in solution")
    raw_moments[idx]
end

function getmomentdist(raw_moments::AbstractVector, rs, momeqs::MomentClosure.MomentEquations, 
                       obs_moments::AbstractMatrix{Int}, ncells)::MultivariateDistribution
    @argcheck size(obs_moments, 2) == length(species(rs))
    nmom = size(obs_moments, 1)

    mean = zeros(nmom)
    cov = zeros(nmom, nmom)

    for i in 1:nmom
        mean[i] = extractmoment(raw_moments, tuple(obs_moments[i,:]...), momeqs)
    end

    for i in 1:nmom
        for j in 1:i
            moment = tuple((obs_moments[j,:] .+ obs_moments[i,:])...)
            cov[i,j] = cov[j,i] = extractmoment(raw_moments, moment, momeqs) - mean[i] * mean[j]
        end
    end

    MvNormal(mean, cov / ncells)
end

##

function logl_moments(params)
    ret = 0.0
    global prob = remake(prob,p=params)
    sol = solve(prob, KenCarp4(), saveat=tt)
    
    dists = try
        getmomentdist(sol, rs, momeqs, [ 0 0 0 0 1; 0 0 0 0 2 ], ncells)
    catch e
        e isa LinearAlgebra.PosDefException && return -Inf
        throw(e)
    end

    for (i, t) in enumerate(tt)
        ret += logpdf(dists[i], obs_moments[i,:])
    end

    ret
end

##

using BlackBoxOptim

searchranges = [ (-3, 3) for i in 1:12 ]
searchranges[5] = (-10000, 10000)

function target(ps)
    -logl_moments(convert_params(ps))
end

res = bboptimize(target; SearchRange=searchranges, NumDimensions=12, MaxSteps=10000)
print(res)

