"""
    MixedMembershipModel

Multilevel mixed membership model for modeling the probability of annullment as a sum of individual judge effects.
"""
struct MixedMembershipModel{T,S,U}
    ys::T
    js::S
    ns::U
end

function MixedMembershipModel(decisions::Vector{Decision})
    ys = (id ∘ outcome).(decisions)
    #js = [SVector(id.(judges(d))...) for d in decisions] # leads to lots of allocations in logdensity evaluation
    js = [id.(judges(d)) for d in decisions]
    ns = length.(js)
    MixedMembershipModel(ys, js, ns)
end

function (problem::MixedMembershipModel)(θ)
    @unpack α, δs, σ = θ
    @unpack ys, js, ns = problem

    loglik = sum(logpdf(Bernoulli(logistic(α + sum(@views δs[j]) / n)), y) for (y, j, n) in zip(ys, js, ns))
    logpri = sum(logpdf(Normal(0, σ), δ) for δ in δs) + logpdf(Normal(0, 1.5), α) + logpdf(Exponential(1), σ)
    loglik + logpri
end

function transformation(problem::MixedMembershipModel)
    as((δs=as(Array, asℝ, maximum(reduce(vcat, problem.js))), α=asℝ, σ=asℝ₊))
end

function predict(problem::MixedMembershipModel, post::DynamicHMCPosterior)
	@unpack ys, js, ns = problem
	
	map(post) do s
		@unpack α, δs, σ = s
		map(zip(js, ns)) do (j, n)
			logistic(α + sum(@views δs[j]) / n)
		end
	end
end
