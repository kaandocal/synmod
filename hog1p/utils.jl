using MAT

function hog1pload(filename; salt="Salt04", exp="Hot1", rep="Rep1")
    datafile = matread(filename)
    tt = dropdims(datafile["Data_Times"], dims=1)[2:end]
    
    experiment = datafile["Exp_Data"][salt][exp][rep]
    
    data_obs = reshape(transpose(experiment["Data"]), (16, 1, size(experiment["Data"], 1)))[2:end,:,:]
    n_cells = dropdims(experiment["N_cells"], dims=2)[2:end]
    
    hist_obs = reshape(transpose(experiment["Data"][:,2:end]), (15, 1, 151))
    
    data_obs = round.(Int, hist_obs .* n_cells)
end

##

function fsp_rhs_hog1p(du, u, p, t)
    aa_fwd = p[1:3]
    aa_bwd = (p[4], p[6], p[7])

    bb = (p[5], 0., 0.)
    ρρ = p[8:11]
    δ = p[12]

    fill!(view(du, :,1), 0.0)

    hog = hog1p(t)

    for i in 1:3
        for j in 1:size(du,1)
            tmp = u[j,i] * aa_fwd[i]
            du[j,i] -= tmp
            du[j,i+1] = tmp
        end

        for j in 1:size(du,1)
            tmp = u[j,i+1] * max(0, aa_bwd[i] + bb[i] * hog)
            du[j,i] += tmp
            du[j,i+1] -= tmp
        end
    end

    for i in 1:4
        for j in 1:size(du,1)-1
            tmp = u[j,i] * ρρ[i]
            du[j,i] -= tmp
            du[j+1,i] += tmp

            tmp = u[j+1,i] * j * δ
            du[j+1,i] -= tmp
            du[j,i] += tmp
        end

        du[end,i] -= u[end,i] * ρρ[i]
    end


    du
end

# For 0.4M NaCLl, Hot1 (WT)
function hog1p(t)
    r1 = 6.9e-5    # Might as well be 0
    r2 = 3.6e-3
    eta = 3.1
    A = 9.3e9
    M = 6.4e-4
    t0 = 190
    
    t = max(0, t - t0)
    
    return A * (exp(r2 * t) / (1 - exp(-r1 * t)) + 1 / M) ^ (-eta)
end

using FileIO
using SparsityDetection, SparseArrays

jacinfo = nothing
try
    global jacinfo = load("data/hog1p_jac_sparsity.jld2")
catch e
    @warn "Unable to load Jacobian info: $e"
end


function compute_jac_sparsity(jacinfo::Nothing, u0, gt_params)
    println("Recomputing jacobian sparsity")
    input = rand(size(u0,1),4)
    output = similar(input)
    sparsity_pattern = jacobian_sparsity(fsp_rhs_hog1p, output, input, gt_params, 200.)
    jac_sparsity = Float64.(sparse(sparsity_pattern))

    save("data/hog1p_jac_sparsity.jld2", Dict("dims" => size(u0), "jac_sparsity" => jac_sparsity))
end

compute_jac_sparsity(jacinfo::Dict, u0, gt_params) = jacinfo["jac_sparsity"]

##


function simulate_fsp!(out::AbstractArray{Int,3}, sol::DESolution)::Array{Int64,3}
    @argcheck size(out,2) == 1 DimensionMismatch

    xx = 0:(size(sol.u[1], 1)-1)
    buf = zeros(size(sol.u[1], 1))

    for j in 1:size(out,1)
        buf .= 0

        for k in 1:4
            buf .+= sol.u[j][:,k]
        end

        weights = pweights(buf)

        sample!(xx, weights, view(out, j, 1, :))
    end
    
    out
end


##

function logl_fsp(params; odefunc, u0, data_obs)
    prob = ODEProblem(odefunc, u0, (0.0, tmax), params)

    sol = solve(prob, KenCarp4(), saveat=rsp.tt)
    return logl_fsp(sol, data_obs=data_obs)
end

function logl_fsp(sol::DESolution; data_obs)
    @argcheck sol.t[1] != 0
    buf = zeros(size(sol.u[1], 1))

    ret = 0.0
    for j in 1:size(data_obs, 1)
        buf .= 0
        for k in 1:4
            buf .+= sol.u[j][:,k]
        end

        for i in 1:size(data_obs,3)
            ret += log(abs(buf[1+data_obs[j,1,i]]))
        end
    end

    ret
end

function logl_fsp_hist(sol::DESolution; hist)
    @argcheck sol.t[1] != 0
    buf = zeros(size(sol.u[1], 1))

    ret = 0.0
    for j in 1:size(data_obs, 1)
        buf .= 0
        for k in 1:4
            buf .+= sol.u[j][:,k]
        end

        arr = log.(abs.(buf)) .* hist[j,1,:]
        ret += sum(filter(!isnan, arr))
    end

    ret
end
