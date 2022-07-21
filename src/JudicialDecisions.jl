
"""
    JudicialDecisions

This package currently has two purposes (which might be refactored into separate packages at a later point): 

- Handling of data on judicial decisions, as issued by the German federal patent court (BPatG).
- Bayesian inference for statistical models of judicial decisions, e.g. mixed membership multilevel models.
"""
module JudicialDecisions

using Reexport: @reexport
@reexport using Statistics

using JSON3
using Dates
using Dictionaries: Dictionary, dictionary, sortkeys
using SplitApplyCombine: group
using StructArrays
using Random
using Distributions
using UnPack
using LogDensityProblems
using TransformVariables
using DynamicHMC
using StatsFuns: logistic

include("datamodel.jl")
include("utils.jl")
include("decisionmodels.jl")

include("models/binomial_groups_model.jl")
include("models/mixed_membership_model.jl")

# data handling (types + methods)
export Outcome, Senate, Judge, Decision
export id, label, senate, outcome, judges, patent, date

# data import
export DataSource, BPatG

# bayesian modeling
export transformation, sample, paramnames
export DynamicHMCPosterior

export BinomialGroupsModel, MixedMembershipModel

end
