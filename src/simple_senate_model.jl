
"""
    SenateModelPosterior

Simple binomial model predicting the probability of annullment by senate.
"""
struct BinomialSenateModel{T}
    ys::Vector{Int}
    ns::Vector{Int}
    senate::Vector{T}
end

function BinomialSenateModel(decisions::Vector{Decision})
    gr = group(id ∘ senate, id ∘ outcome, decisions)
    ys = map(sum, gr) |> sortkeys
    ns = map(length, gr) |> sortkeys
    BinomialSenateModel(collect(ys), collect(ns), collect(keys(ns)))
end

function (problem::BinomialSenateModel)(θ)
    @unpack αs, μ, σ = θ
    @unpack ys, ns, senate = problem
    loglik = sum(logpdf(Binomial(n, logistic.(α)), y) for (n, α, y) in zip(ns, αs, ys))
    logpri = sum(logpdf(Normal(μ, σ), α) for α in αs) + logpdf(Normal(0, 1), μ) + logpdf(Exponential(1), σ)
    loglik + logpri
end

function transformation(problem::BinomialSenateModel)
    as((αs=as(Array, asℝ, length(problem.senate)), μ=asℝ, σ=asℝ₊))
end