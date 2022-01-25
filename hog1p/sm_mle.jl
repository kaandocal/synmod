using Catalyst
using DifferentialEquations, DiffEqBase
#using BlackBoxOptim
using AdvancedMH, MCMCChains
using LinearAlgebra
using Sundials

include("nbmixture.jl")
include("synmod.jl")

include("hog1p/utils.jl")

##

data_obs = hog1pload("data/Data_Gregor_04_27_2012.mat")
data_obs = Array{Float64}(data_obs)

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

u0 = zeros(Nmax+1, 4)
u0[1,1] = 1.

jac_sparsity = compute_jac_sparsity(jacinfo, u0, gt_params)
odefunc = ODEFunction{true}(fsp_rhs_hog1p, jac_prototype=jac_sparsity)
prob = ODEProblem(odefunc, u0, (0.0, tmax), gt_params)

##

samplesize = 10000
ncomps = 4

data_sim = zeros(Int64, length(tt), 1, samplesize);

function logl_synmod_parts(params)
    global prob = remake(prob,p=params)
    sol = solve(prob, KenCarp4(), saveat=tt)
    simulate_fsp!(data_sim, sol)
   
    syn = SyntheticModelPS(tt, [ncomps])
    fit!(syn, data_sim, maxiter=1000)
    
    xx = 0:Nmax

    ret = zeros(length(tt))
    for j in 1:length(tt)
        mixture = syn.mixtures[1,j]
        lpj = sum(logpdf(mixture, xx) .* data_obs[j,1,:])
        ret[j] = lpj
    end

    ret
end

function logl_synmod(params)
    sum(logl_synmod_parts(params))
end


##

using BlackBoxOptim

 
searchranges = [ (-3, 3) for i in 1:12 ]
searchranges[5] = (-10000, 10000)

function target(ps)
    -logl_synmod(convert_params(ps))
end

res = bboptimize(target; SearchRange=searchranges, NumDimensions=12, MaxSteps=25000)
print(res)
