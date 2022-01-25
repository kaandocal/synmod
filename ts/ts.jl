    using Catalyst
    using StaticArrays
    using DifferentialEquations

    using CSV, DataFrames

    data_obs = Matrix(CSV.read("data/ts/data_obs.csv", DataFrame))
    data_obs = reshape(data_obs, 8, 2, size(data_obs, 2))

    ##

    @parameters σ_b1 σ_u1 ρ_m1 ρ_p1
    @parameters σ_b2 σ_u2 ρ_m2 ρ_p2

    gt_params = [ 12., 6., 15., 30., 1.5, 7., 1.7, 60.]

    ranges = [ 0.0  40.
               0.0  20.
               0.0  50.
               0.0  100.
               0.0  8.
               0.0  25.
               0.0  10.
               0.0  200.]

    prior = Product(Uniform.(ranges[:,1], ranges[:,2]))

    ##

    rn = @reaction_network begin
        (σ_b1, σ_u1), Gu_A + Pn_B <--> Gb_A
        ρ_m1, Gu_A --> Gu_A + Mn_A
        12, Mn_A --> Mc_A
        3, Mc_A --> 0
        ρ_p1, Mc_A --> Mc_A + Pc_A
        (4, 4), Pc_A <--> Pn_A
        2, Pc_A --> 0
        (σ_b2, σ_u2), Gu_B + Pn_A <--> Gb_B
        ρ_m2, Gu_B --> Gu_B + Mn_B
        16, Mn_B --> Mc_B
        1, Mc_B --> 0
        ρ_p2, Mc_B --> Mc_B + Pc_B
        (4, 4), Pc_B <--> Pn_B
        3, Pc_B --> 0
    end σ_b1 σ_u1 ρ_m1 ρ_p1 σ_b2 σ_u2 ρ_m2 ρ_p2

    tt = 1.:1:8

    u0 = @SArray [ 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ]

    ##

    dprob = DiscreteProblem(rn, u0, (0.0, last(tt)), gt_params)
    jprob = JumpProblem(rn, dprob, Direct(), save_positions=(false, false))

function simulate!(out, ps)
    N = size(out, 3)
    _jprob = remake(jprob, p=ps)

    Threads.@threads for i in 1:N
        sol = solve(_jprob, SSAStepper(), saveat=tt)
        out[:,1,i] .= map(x -> x[6], sol.u[2:end])
        out[:,2,i] .= map(x -> x[12], sol.u[2:end])
    end

    out
end

data_obs_stim = Matrix(CSV.read("data/ts/data_obs_stim.csv", DataFrame))
data_obs_stim = reshape(data_obs_stim, 8, 2, size(data_obs_stim, 2))


const stimulus = [ 20., 20. ]

cb_condition(u,t,int) = t in (tt[1:2:end] .+ 0.001)
function cb_affect!(integrator)
    integrator.u[6] += stimulus[1]
    integrator.u[12] += stimulus[2]
end

stim_cb = DiscreteCallback(cb_condition, cb_affect!)

function stimulate!(out, ps)
    N = size(out, 3)
    _jprob = remake(jprob, p=ps)

    Threads.@threads for i in 1:N
        sol = solve(_jprob, SSAStepper(), saveat=tt, tstops=tt)
        out[:,1,i] .= map(x -> x[6], sol.u[2:end])
        out[:,2,i] .= map(x -> x[12], sol.u[2:end])
    end

    out
end

