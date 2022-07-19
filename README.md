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
One of the simplest models is a Binomial model for the probability that a given senate will anull a patent (i.e., cross-senate variation):

$$
y_s \sim \textrm{Binomial}(n_s, p_s), \;\; \textrm{for} \; i=1, \ldots, S
$$

where $S$ refers to the total number of senates.

Here is how we would use the implementation of this model in the package:


```julia
problem = BinomialSenateModel(decisions);
post = sample(problem, 1000)
DynamicHMCPosterior with 1000 samples and parameters (:ps,)

julia> mean(post.ps)
8-element Vector{Float64}:
 0.6983246848950473
 0.7702156094539311
 0.7642127632354627
 0.7971603887783285
 0.7333720201613461
 0.7651470656631522
 0.8560307824560833
 0.7171744109160901
```

