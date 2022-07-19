
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
    @unpack ps = θ
    @unpack ys, ns, senate = problem

    loglik = sum(logpdf(Binomial(n, p), y) for (n, p, y) in zip(ns, ps, ys))
    logpri = sum(logpdf(Beta(2,2), p) for p in ps)

    loglik + logpri
end

transformation(problem::BinomialSenateModel) = as((ps=as(Array, as𝕀, length(problem.senate)),))