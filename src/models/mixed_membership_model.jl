"""
    MixedMembershipModel

Multilevel mixed membership model modeling the probability of annullment as a sum of individual judge effects.
"""
struct MixedMembershipModel
    ys::Vector{Bool}
    js::Vector{Vector{Int}}
    ns::Vector{Int}
end

function MixedMembershipModel(decisions::Vector{Decision})
    ys = (id ∘ outcome).(decisions)
    js = [id.(judges(d)) for d in decisions]
    ns = length.(js)
    MixedMembershipModel(ys, js, ns)
end

function (problem::MixedMembershipModel)(θ)
    @unpack α, δs, μ, σ = θ
    @unpack ys, js, ns = problem
    loglik = sum(logpdf(Bernoulli(logistic(α + sum(@views δs[j]))), y) for (y, j, n) in zip(ys, js, ns))
    logpri = sum(logpdf(Normal(μ, σ), δ) for δ in δs) + logpdf(Normal(0, 1), μ) + logpdf(Exponential(1), σ)
    loglik + logpri
end

function transformation(problem::MixedMembershipModel)
    as((δs=as(Array, asℝ, maximum(reduce(vcat, problem.js))), α=asℝ, μ=asℝ, σ=asℝ₊))
end
