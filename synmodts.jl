using Distributions
using StatsBase
using HMMBase

mutable struct SyntheticModelTS
    tt::Vector{Float64}
    hmms::Vector{TimeVaryingHMM{NegativeBinomial,Float64}}
end

function SyntheticModelTS(tt::AbstractVector{<:Number}, nstates::AbstractVector{Int}) 
    d = length(nstates)
    T = length(tt)
    hmms = [ TimeVaryingHMM{NegativeBinomial,Float64}(ones(nstates[i]) ./ nstates[i],
                                                      ones(nstates[i], nstates[i], T-1) ./ nstates[i], 
                                                      fill(NegativeBinomial(0.001, 0.999), nstates[i], T))
             for i in 1:d ]
    
    SyntheticModelTS(tt, hmms)
end

function fitmixtures!(syn::SyntheticModelTS, ::Type{NegativeBinomialMixture}, 
                      data::AbstractArray{Int,3}; kwargs...)
    for (i, hmm) in enumerate(syn.hmms)
        fitmixtures!(hmm, NegativeBinomialMixture, @view data[:,i,:]; kwargs...) || return false
    end
   
    true
end

function fitmixture!(hmm::TimeVaryingHMM, t::Int, data::AbstractVector{Int}; kwargs...)
    ncomps = size(hmm.B, 1)

    mixture = fit(NegativeBinomialMixture, ncomps, data; kwargs...)
        
    isvalidmixture(mixture) || return false
    
    hmm.B[:,t] .= mixture.components

    if t == 1
        hmm.a .= mixture.prior.p
    else
        for k in 1:ncomps
            hmm.A[k,:,t-1] .= mixture.prior.p
        end
    end

    true
end

function fitmixtures!(hmm::TimeVaryingHMM, ::Type{NegativeBinomialMixture}, 
                      data::AbstractMatrix{Int}; threads=true, kwargs...)
    T = size(hmm.B, 2)

    if !threads
        for t in 1:T
            fitmixture!(hmm, t, @view data[t,:]; kwargs...) || return false
        end
    else
        Threads.@threads for t in 1:T
            fitmixture!(hmm, t, @view data[t,:]; kwargs...) || return false
        end
    end
    
    true
end

function fithmm!(syn::SyntheticModelTS, data::AbstractArray{Int,3}; kwargs...)
    for (i, hmm) in enumerate(syn.hmms)
        fit_mle!(hmm, (@view data[:,i,:]); kwargs...)
    end
    
    syn
end

function StatsBase.fit!(syn::SyntheticModelTS, data::AbstractArray{Int,3}; 
                        ts=true, hmm_kwargs=Dict(), mixture_kwargs=Dict())
    fitmixtures!(syn, NegativeBinomialMixture, data; mixture_kwargs...) || syn
    ts && fithmm!(syn, data; hmm_kwargs...)
    
    syn
end

function StatsBase.loglikelihood(syn::SyntheticModelTS, data::AbstractArray{Int,3})::Float64
    sum(loglikelihood(hmm, @view data[:,i,:]) for (i, hmm) in enumerate(syn.hmms))
end
