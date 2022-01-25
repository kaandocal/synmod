function getmomentdist(sol::DESolution, rn, momeqs::MomentClosure.MomentEquations, 
                       obs_moments::AbstractMatrix{Int}, ncells::AbstractVector)
    @argcheck size(obs_moments, 2) == length(species(rn))

    ret = Array{MultivariateDistribution}(undef, length(sol.t))
    for (i, t) in enumerate(sol.t)
        ret[i] = getmomentdist(sol.u[i], rn, momeqs, obs_moments, ncells[i])
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

function getmomentdist(raw_moments::AbstractVector, rn, momeqs::MomentClosure.MomentEquations, 
                       momentlist::AbstractMatrix{Int}, ncells)::MultivariateDistribution
    @argcheck size(momentlist, 2) == length(species(rn))
    nmom = size(momentlist, 1)

    mean = zeros(nmom)
    cov = zeros(nmom, nmom)

    for i in 1:nmom
        mean[i] = extractmoment(raw_moments, tuple(momentlist[i,:]...), momeqs)
    end

    for i in 1:nmom
        for j in 1:i
            moment = tuple((momentlist[j,:] .+ momentlist[i,:])...)
            cov[i,j] = cov[j,i] = extractmoment(raw_moments, moment, momeqs) - mean[i] * mean[j]
        end
    end

    MvNormal(mean, cov / ncells + 1e-6 * I)
end
