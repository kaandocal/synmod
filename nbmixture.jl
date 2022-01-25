using Distributions, StatsBase
using SpecialFunctions
using Printf

##

struct NegativeBinomialMixture <: AbstractMixtureModel{Univariate,Discrete,NegativeBinomial{Float64}}
    components::Vector{NegativeBinomial{Float64}}
    prior::Categorical{Float64,Vector{Float64}}
end

Distributions.ncomponents(mixture::NegativeBinomialMixture) = length(mixture.components)
Distributions.components(mixture::NegativeBinomialMixture) = mixture.components
Distributions.component(mixture::NegativeBinomialMixture, k::Int) = mixture.components[k]

Distributions.probs(mixture::NegativeBinomialMixture) = mixture.prior.p

function Base.show(io::IO, d::NegativeBinomialMixture)
    K = ncomponents(d)
    pr = probs(d)
    println(io, "NegativeBinomialMixture(K = $K)")
    Ks = min(K, 8)
    for i = 1:Ks
        @printf(io, "components[%d] (prior = %.4f): ", i, pr[i])
        println(io, component(d, i))
    end
    if Ks < K
        println(io, "The rest are omitted ...")
    end
end

function sumlogpdf(dist::Distribution, samples::AbstractVector, weights::AbstractVector)::Float64
    ret = 0.0
    
    for (sample, weight) in zip(samples, weights)
        ret += logpdf(dist, sample) * weight
    end
    
    ret
end

function createhist(samples::AbstractVector{Int})
    samples_unique = unique(samples)
    weights = FrequencyWeights([ count(x -> x == s, samples) for s in samples_unique ])
    
    return (samples_unique, weights)
end

isvalidmixture(mixture::NegativeBinomialMixture) = isapprox(sum(mixture.prior.p), 1)

function initmixture!(mixture::NegativeBinomialMixture, samples::AbstractVector{Int}, weights::AbstractWeights)::NegativeBinomialMixture
    ncomps = ncomponents(mixture)
    mixture.prior.p .= 1 / ncomps
    
    if all(iszero.(samples))
        #@warn "Fitting mixture to all zero datapoints"
        for i in 1:ncomps
            mixture.components[i] = NegativeBinomial(0.001, 0.999)
        end
        
        return mixture
    end
    
    ps = (1:ncomps) / (ncomps + 1)
    
    qs = quantile(samples, weights, ps)
    var_per_comp = (var(samples, weights) + 0.1) / ncomps
    
    means = abs.(qs .+ sqrt(0.1 * var_per_comp) .* randn(ncomps))
    
    for (i, m) in enumerate(means)
        mixture.components[i] = negbinfrommoments(m, var_per_comp)
    end
    
    mixture
end

function negbinfrommoments(mean::Real, var::Real)::NegativeBinomial{Float64}
    p = clamp(mean / var, 1e-3, 1 - 1e-3)
    r = mean * p / (1 - p)
    
    NegativeBinomial{Float64}(r, p)
end

function StatsBase.fit!(mixture::NegativeBinomialMixture, samples::AbstractVector{Int}; 
                        kwargs...)::NegativeBinomialMixture
    fit!(mixture, createhist(samples)...; kwargs...)
end

function StatsBase.fit(::Type{NegativeBinomialMixture}, ncomps::Int, 
                       samples::AbstractVector{Int}; kwargs...)::NegativeBinomialMixture
    mixture = NegativeBinomialMixture([ negbinfrommoments(0.001, 0.999) for i in 1:ncomps ],
                                        Categorical(ones(ncomps) ./ ncomps))
    
    fit!(mixture, samples; kwargs...)
end

function StatsBase.fit!(mixture::NegativeBinomialMixture, samples::AbstractVector{Int}, 
                        weights::AbstractWeights; maxtries=5, kwargs...)
    for i in 1:maxtries
        initmixture!(mixture, samples, weights)
        attempt_fit!(mixture, samples, weights; kwargs...)
        
        isvalidmixture(mixture) && break
    end
    
    isvalidmixture(mixture) || @warn "Unable to fit NB mixture" 
    
    mixture
end

function attempt_fit!(mixture::NegativeBinomialMixture, samples::AbstractVector{Int}, 
                      weights::AbstractWeights; 
                      maxiter=250, es_min=50, es_thresh=0.0, losses=nothing)::NegativeBinomialMixture
    all(samples .>= 0) || error("Cannot fit mixture to negative samples")
    
    ncomps = ncomponents(mixture)
    
    if all(iszero.(samples))
        @warn "Fitting mixture to all zero datapoints"
        
        for i in 1:ncomps
            mixture.components[i] = NegativeBinomial(0.001, 0.999)
        end
        
        mixture.prior.p .= 1 / ncomps
        return mixture
    end
    
    last_loss::Float64 = Inf
    buffer = Array{Float64}(undef, ncomps, length(samples), 2)
        
    for i in 1:maxiter
        try
            emstep!(mixture, samples, weights, buffer)
        catch err
            err isa ArgumentError || rethrow()
            @warn "Invalid NB mixture during fit, stopping early"
            mixture.prior.p .= NaN
            return mixture
        end

        losses != nothing && push!(losses, -sumlogpdf(mixture, samples, weights))
    
        if i > es_min && i % 10 == 0
            loss::Float64 = losses === nothing ? -sumlogpdf(mixture, samples, weights) : last(losses)
            (abs(loss - last_loss) <= es_thresh) && break
            last_loss = loss
        end
    end
    
    mixture
end

function getlatents!(out::AbstractMatrix{Float64}, mixture::NegativeBinomialMixture, samples::AbstractVector{Int})
    for (i, comp) in enumerate(mixture.components)
        for (k, sample) in enumerate(samples)
            out[i,k] = pdf(comp, sample)
        end
    end
    
    for k in 1:length(samples)
        s = sum(@view out[:,k])
        
        for i in 1:length(mixture.components)
            out[i,k] /= s + 1e-30
        end
    end
    
    out
end

function getdeltas!(out::AbstractMatrix{Float64}, mixture::NegativeBinomialMixture, samples::AbstractVector{Int})
    for (i, comp) in enumerate(mixture.components)
        dig = digamma(comp.r + 1e-12)
        for (k, sample) in enumerate(samples)
            out[i,k] = comp.r * (digamma(comp.r + sample + 1e-12) - dig)
        end
    end
    
    out
end


function emstep!(mixture::NegativeBinomialMixture, samples::AbstractVector{Int}, weights::AbstractWeights, buffer::AbstractArray{Float64,3};
                 p_eps=1e-3, r_eps=1e-3)
    zz = @view buffer[:,:,1]
    deltas = @view buffer[:,:,2]
        
    getlatents!(zz, mixture, samples)
    getdeltas!(deltas, mixture, samples)
    
    for (i, comp) in enumerate(mixture.components)
        sum_zz_delta = 0.0
        sum_zz_samples = 0.0
        sum_zz = 0.0
        
        for j in 1:length(samples)
            sum_zz_delta += zz[i,j] * deltas[i,j] * weights[j]
            sum_zz_samples += zz[i,j] * samples[j] * weights[j]
            sum_zz += zz[i,j] * weights[j]
        end
        
        lambda = sum_zz_delta / sum_zz 
        beta = 1 - 1 / (1 - comp.p + 1e-12) - 1 / (log(comp.p + 1e-12) + 1e-12)
        
        theta = beta * sum_zz_delta / (sum_zz_samples - (1 - beta) * sum_zz_delta)
        
        theta = clamp(theta, p_eps, 1 - p_eps)
        
        new_r = -lambda / log(theta)
        new_r = max(new_r, r_eps)
        
        new_p = theta
        
        mixture.components[i] = NegativeBinomial(new_r, new_p)
        
        mixture.prior.p[i] = sum_zz
    end
    
    mixture.prior.p ./= sum(mixture.prior.p)
    
    mixture
end
