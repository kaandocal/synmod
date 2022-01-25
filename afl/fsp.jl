using Catalyst
using AdvancedMH, MCMCChains
using LinearAlgebra, SparseArrays
using Expokit
using FiniteStateProjection

include("nbmixture.jl")
include("synmod.jl")

experiment = "nfl"
include("afl/afl.jl")

##

# 150 for PFL, 200 for NFL
Nmax = 200

sys = FSPSystem(rn) 

u0_fsp = zeros(2, Nmax+1)
u0_fsp[(u0 .+ 1)...] = 1.

##

function logl_fsp_ts(params; tol=1e-9) 
    A = convert(SparseMatrixCSC, sys, size(u0_fsp), params, 0.0)
    ut_fsp0 = reshape(expmv(tt[1], A, vec(u0_fsp), tol=tol), size(u0_fsp))

    @show params
    N = size(data_obs, 2)
    rets = zeros(N)

    for j in 1:size(data_obs,2)
        y = data_obs[1,j]
        rets[j] = log(abs(ut_fsp0[1,y+1] + ut_fsp0[2,y+1]))
    end

    Threads.@threads for j in 1:size(data_obs,2)
        ut_fsp = copy(ut_fsp0)

        for i in 2:length(tt)
            # Filtering: update the solution of the FSP
            # conditioned on the last observation
            y_old = data_obs[i-1,j]
            u0_fsp = similar(ut_fsp)
            u0_fsp .= 0
            u0_fsp[:,y_old+1] .= ut_fsp[:,y_old+1] / sum(ut_fsp[:,y_old+1])

            expmv!(vec(ut_fsp), tt[i] - tt[i-1], A, vec(u0_fsp), tol=tol)

            y = data_obs[i,j]
            rets[j] += log(abs(ut_fsp[1,y+1] + ut_fsp[2,y+1]))
        end
    end

    sum(rets)
end

function logl_fsp_ps(params) 
    A = convert(SparseMatrixCSC, sys, size(u0_fsp), params, 0.0)
    ut_fsp = reshape(expmv(tt[1], A, vec(u0_fsp), tol=1e-9), size(u0_fsp))

    ret = 0
    for j in 1:size(data_obs,2)
        y = data_obs[1,j]
        ret += log(abs(ut_fsp[1,y+1] + ut_fsp[2,y+1]))
    end

    up_fsp = copy(u0_fsp)

    for i in 2:length(tt)
        temp = up_fsp
        up_fsp = ut_fsp
        ut_fsp = temp

        expmv!(vec(ut_fsp), tt[i] - tt[i-1], A, vec(up_fsp), tol=1e-9)

        for j in 1:size(data_obs,2)
            y = data_obs[i,j]
            ret += log(abs(ut_fsp[1,y+1] + ut_fsp[2,y+1]))
        end
    end

    ret
end

##

insupport(ps) = all(ranges[:,1] .<= ps .<= ranges[:,2])
pdensity(ps) = pdf(prior, ps) != 0 ? logl_fsp_ts(ps) + logpdf(prior, ps) : -Inf

model = DensityModel(pdensity)

D = size(ranges, 1)
trker_std = 0.025 .* sqrt.(diag(cov(prior)))
spl = RWMH(MvNormal(zeros(D), diagm(trker_std .^ 2)))

##

init_params = rand(prior)
chain = sample(model, spl, 50000; init_params=init_params, chain_type=Chains)
CSV.write("data/afl/posterior_$(experiment)_fsp.csv", Tables.table(chain.value.data[:,:,1]))
