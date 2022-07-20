# JudicialDecisions

This package currently has two purposes (which might be refactored into separate packages at a later point): 
- Handling of data on judicial decisions, as issued by the German federal patent court (BPatG).
- Bayesian inference for statistical models of judicial decisions, e.g. mixed membership multilevel models.

# Data handling & import

`JudicialDecisions.loaddata(BPatG(), dir)` parses files in `dir`, which are expected to be `.jsonl` representations
of decisisons made by the German federal patent court, into an array with elements of type `Decision`:

```julia
julia> decisions = JudicialDecisions.loaddata(BPatG(), "../data/json")
1565-element Vector{Decision}:
 1 Ni 4/99 (EU)
 1 Ni 6/99(EU)
 1 Ni 8/99
 ⋮
 7 Ni 55/19 (EP)
 7 Ni 60/19 (EP)
 5 Ni 27/19 (EP)
```

Printing out a single decision showcases its metadata:

```julia
julia> decisions[1]
Ruling 1 Ni 4/99 (EU) on EP0389008
Date of decision: 26 September, 2000
Decided by: 1. Senate (Hacker, K Vogel, Henkel, W Maier, van Raden)
Outcome: partially annulled
```
There are accessor functions for most information in a `Decision`, e.g.:

```julia
julia> outcome(decisions[1])
Outcome(1, "partially annulled")
```

# Bayesian models for judicial decision making

The package contains a series of statistical model implementations for modeling aspects of the decision process.
We start with a simple hierarchical Binomial model for the probability that a given senate will nullify a patent (i.e., a model of cross-senate variation):

$$
\begin{align}
y_s &\sim \textrm{Binomial}(n_s, p_s), \text{ for } s=1, \ldots, S \\
p_s &= \textrm{logit}^{-1}(\alpha_s) \\
\alpha_s &\sim \textrm{Normal}(\mu, \sigma) \\
\mu &\sim \textrm{Normal}(0, 1) \\
\sigma & \sim \textrm{Exponential}(1) \\
\end{align}
$$

where $S$ refers to the total number of senates.

Here is how we would use the implementation of this model in the package:


```julia
julia> problem = BinomialGroupsModel(decisions; groupfun=id ∘ senate);
julia> post = sample(problem, 1000)
DynamicHMCPosterior with 1000 samples and parameters (:αs, :μ, :σ)

julia> mean(post.αs)
8-element Vector{Float64}:
 1.1533265420437266
 1.2302096660141322
 1.2042245345786518
 1.2937742490215314
 1.1191711839066207
 1.215801205986333
 1.3576725865081927
 1.166707942601264
```

