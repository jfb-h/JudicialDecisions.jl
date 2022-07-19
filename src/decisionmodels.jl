abstract type AbstractDecisionModel end


"""
    transformation(problem)

Transform parameters to the unconstraint Real numbers. 
Implement this using `TransformVariables.as` on a per-model basis.
"""
function transformation(problem)
    throw(ArgumentError("Not implemented for $(typeof(problem))"))
end


abstract type SamplingAlgorithm end
struct NUTS end


abstract type AbstractPosterior end

"""
    DynamicHMCPosterior

struct representing the posterior distribution of a model's parameters sampled with DynamicHMC.
Also used for dispatch in the `sample` method.
"""
struct DynamicHMCPosterior{T,S}
    post::T
    stat::S
end

stats(x::DynamicHMCPosterior) = getfield(x, :stat)
_post(x::DynamicHMCPosterior) = getfield(x, :post)

paramnames(x::DynamicHMCPosterior) = keys(first(_post(x)))

Base.length(x::DynamicHMCPosterior) = length(_post(x))
Base.getindex(x::DynamicHMCPosterior, key) = getindex(_post(x), key)
Base.getproperty(x::DynamicHMCPosterior, f::Symbol) = getproperty(_post(x), f)

Base.iterate(x::DynamicHMCPosterior) = iterate(_post(x))
Base.iterate(x::DynamicHMCPosterior, state) = iterate(_post(x), state)

function Base.show(io::IO, ::MIME"text/plain", p::DynamicHMCPosterior)
    #compact = get(io, :compact, false)
    params = paramnames(p)
    print(io, "DynamicHMCPosterior with $(length(p)) samples and parameters $params")
end


"""
    sample(DynamicHMCPosterior(), problem, iter)

Sample from the posterior distribution of `problem` with the sampling 
algorithm specified by the first argument, taking `iter` samples.
If the first argument is omitted, NUTS via `DynamicHMC`` is used by default.
"""
function sample(::NUTS, problem, iter; backend=:ForwardDiff)
    t = transformation(problem)
    ℓ = TransformedLogDensity(t, problem)
    ∇ℓ = ADgradient(backend, ℓ)
    r = mcmc_with_warmup(Random.GLOBAL_RNG, ∇ℓ, iter)
    post = StructArray(TransformVariables.transform.(t, r.chain))
    stat = (tree_statistics=r.tree_statistics, κ=r.κ, ϵ=r.ϵ)
    DynamicHMCPosterior(post, stat)
end

sample(problem, iter; kwargs...) = sample(NUTS(), problem, iter; kwargs...)