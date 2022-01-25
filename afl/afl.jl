using Catalyst
using StaticArrays
using Distributions
using DifferentialEquations

using CSV, DataFrames

load_data(name) = Matrix(CSV.read("data/afl/data_obs_$(name).csv", DataFrame))'

experiments = ["pfl", "nfl"]

@assert experiment in experiments

data_obs_all = Dict(experiment => load_data(experiment) for experiment in experiments)

data_obs = data_obs_all[experiment]

##

@parameters σ_u σ_b ρ_u ρ_b b

gt_params_all = Dict("pfl" => [ 0.01, 0.004, 10., 35. ],
                     "nfl" => [  0.1, 0.001, 13, 0., 3. ])
 
ranges_all = Dict("pfl" => [ 0. 0.2
                             0. 0.02
                             0. 50
                             0. 50 ],
                  "nfl" => [ 0. 0.5
                             0. 0.01
                             0. 50
                             0. 10
                             0. 5 ]
                             )

gt_params = gt_params_all[experiment]
ranges = ranges_all[experiment]

prior = Product(Uniform.(ranges[:,1], ranges[:,2]))

##

function add_bursty_reactions!(sys, ρ_u, ρ_b, b; Nmax=25)
    Gterm = sys.states[1]
    Pterm = sys.states[2]
    
    for j in 2:Nmax
        rxu = Reaction{Any, Int}((ρ_u * (b^j) / (1 + b)^(j+1)).val, 
                       sys.eqs[3].substrates, sys.eqs[3].products, 
                       [1], [1,j], Pair{Any,Int}[ Pterm => j ], 
                       false)

        rxb = Reaction{Any, Int}((ρ_b * (b^j) / (1 + b)^(j+1) * (1 - Gterm)).val,
                       sys.eqs[4].substrates, sys.eqs[4].products, 
                       Int[], [j], Pair{Any,Int}[ Pterm => j ], 
                       false)

        addreaction!(sys, rxu)
        addreaction!(sys, rxb)
    end
    
    sys
end

##

if !(experiment in ["nfl"])
    rn = @reaction_network begin
        σ_u * (1 - G), 0 --> G + P
        σ_b, G + P --> 0
        ρ_u, G --> G + P
        ρ_b * (1 - G), 0 --> P
        1, P --> 0
    end σ_u σ_b ρ_u ρ_b     
else
    rn = @reaction_network begin
        σ_u * (1 - G), 0 --> G + P
        σ_b, G + P --> 0
        ρ_u * b / (1+b)^2, G --> G + P
        ρ_b * b / (1+b)^2 * (1 - G), 0 --> P
        1, P --> 0
    end σ_u σ_b ρ_u ρ_b b

    add_bursty_reactions!(rn, ρ_u, ρ_b, b)
end


tt_all = Dict("pfl" => 4:4.:16,
              "nfl" => 4:4.:16)

tt = tt_all[experiment]

u0 = [ 1, 0 ]

##

solvers_all = Dict("pfl" => Direct(), "nfl" => DirectCR())

dprob = DiscreteProblem(rn, u0, (0.0, last(tt)), gt_params)
jprob = JumpProblem(rn, dprob, solvers_all[experiment], save_positions=(false,false))

function simulate!(margmap, out, ps)
    N = size(out, 3)
    _jprob = remake(jprob, p=ps)

    for i in 1:N
        sol = solve(_jprob, SSAStepper(), saveat=tt)
        out[:,:,i] .= map(margmap, sol.u[2:end])
    end

    out
end

