using Distributions
using StatsBase
using ArgCheck

mutable struct SyntheticModelPS
    tt::Vector{Float64}
    mixtures::Matrix{NegativeBinomialMixture}
end

function SyntheticModelPS(tt::AbstractVector{<:Number}, ncomps::AbstractVector{Int}) 
    mixtures = [ NegativeBinomialMixture(fill(NegativeBinomial(0.001, 0.999), ncomps[i]), 
                                         Categorical(ncomps[i])) 
                for i in 1:length(ncomps), j in 1:length(tt) ]
    
    SyntheticModelPS(tt, mixtures)
end

function StatsBase.fit!(syn::SyntheticModelPS, data::AbstractArray{Int,3}; threads=true, kwargs...)
    d, T = size(syn.mixtures)

    @argcheck d == size(data,2)
    @argcheck T == size(data,1)

    if !threads
        for i in 1:d
            for t in 1:T
                fit!(syn.mixtures[i,t], @view data[t,i,:]; kwargs...) 
                isvalidmixture(syn.mixtures[i,t]) || return false
            end
        end
    else
        Threads.@threads for i in 1:d
            for t in 1:T
                fit!(syn.mixtures[i,t], @view data[t,i,:]; kwargs...) 
                isvalidmixture(syn.mixtures[i,t]) || return false
            end
        end
    end
    
    return true
end

function StatsBase.loglikelihood(syn::SyntheticModelPS, data::AbstractArray{Int,3})
    d, T = size(syn.mixtures)

    @argcheck d == size(data,2)
    @argcheck T == size(data,1)

    sum(sum.(logpdf(syn.mixtures[i,t], (@view data[t,i,:])) for i in 1:d, t in 1:T))
end
