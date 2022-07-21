abstract type AbstractDecisionModel end
abstract type AbstractInferenceAlgorithm end
abstract type AbstractPosterior end

"""
    transformation(problem)

Transform parameters to the unconstraint Real numbers. 
Implement this using `TransformVariables.as` on a per-model basis.
"""
function transformation(problem)
    throw(ArgumentError("Not implemented for $(typeof(problem))"))
end


"""
    DynamicHMCPosterior

struct representing the posterior distribution of a model's parameters sampled with DynamicHMC.
Also used for dispatch in the `sample` method.
"""
struct DynamicHMCPosterior{T,S} <: AbstractPosterior
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


struct NUTS <: AbstractInferenceAlgorithm end

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
    r = mcmc_with_warmup(Random.GLOBAL_RNG, ∇ℓ, iter; reporter=ProgressMeterReport())
    post = StructArray(TransformVariables.transform.(t, r.chain))
    stat = (tree_statistics=r.tree_statistics, κ=r.κ, ϵ=r.ϵ)
    DynamicHMCPosterior(post, stat)
end

sample(problem, iter; kwargs...) = sample(NUTS(), problem, iter; kwargs...)
sample(problem, iter, chains; kwargs...) = error("Sampling with multiple chains not implemented yet.")


"""
    predict(problem, post)

Perform posterior prediction base on posterior distribution `post` over the data in `problem`.
"""
function predict(problem::AbstractDecisionModel, post::AbstractPosterior) 
    throw(ArgumentError("Not implemented yet for $(typeof(problem)). Needs to be implemented on a per-model basis."))
end

# """
#     checkconvergence(post)

# Check effective sample sizes and convergence statistics for the NUTS algorithm.
# """
# function checkconvergence(post::DynamicHMCPosterior)
#     ess = vec(mapslices(MCMCDiagnostics.effective_sample_size, DynamicHMC.position_matrix(_post(post)); dims=2))
#     tree_stats = DynamicHMC.Diagnostics.summarize_tree_statistics(stats(post).tree_statistics)
#     (;ess, tree_stats)
# end