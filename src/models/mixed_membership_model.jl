"""
    MixedMembershipModel

Multilevel mixed membership model for modeling the probability of annullment as a sum of individual judge effects.
"""
struct MixedMembershipModel{T,S,U}
    ys::T
    js::S
    ns::U
    N::Int
end

function MixedMembershipModel(decisions::Vector{Decision})
    ys = (id ∘ outcome).(decisions)
    #js = [SVector(id.(judges(d))...) for d in decisions] # leads to lots of allocations in logdensity evaluation
    js = [id.(judges(d)) for d in decisions]
    ns = length.(js)
    N = maximum(reduce(vcat, js))
    MixedMembershipModel(ys, js, ns, N)
end

function (problem::MixedMembershipModel)(θ)
    @unpack α, zs, σ = θ
    @unpack ys, js, ns, N = problem

    loglik = sum(logpdf(Bernoulli(logistic(α + @views sum(x -> x*σ, zs[j]) / n)), y) for (y, j, n) in zip(ys, js, ns))
    logpri = logpdf(MvNormal(N, 1), zs) + logpdf(Normal(0, 1.5), α) + logpdf(Exponential(0.5), σ)
    loglik + logpri
end

function transformation(problem::MixedMembershipModel)
    as((zs=as(Array, asℝ, maximum(reduce(vcat, problem.js))), α=asℝ, σ=asℝ₊))
end

function predict(problem::MixedMembershipModel, post::DynamicHMCPosterior)
	@unpack ys, js, ns = problem
	
	map(post) do s
		@unpack α, zs, σ = s
		map(zip(js, ns)) do (j, n)
			logistic(α + sum(x -> x*σ, @views zs[j]) / n)
		end
	end
end
