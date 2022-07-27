
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
using CSV
using Dates
using Dictionaries: Dictionary, dictionary, sortkeys
using SplitApplyCombine: group
using StructArrays
using Random
using LinearAlgebra
using Distributions
using UnPack
using LogDensityProblems
using TransformVariables
using DynamicHMC
import ReverseDiff
using MCMCDiagnostics
using StatsFuns: logistic
using StatsBase: countmap
using CairoMakie; set_theme!(theme_light())

include("datamodel.jl")
include("utils.jl")
include("decisionmodels.jl")

include("models/binomial_groups_model.jl")
include("models/mixed_membership_model.jl")
include("models/multi_mixed_membership_model.jl")

include("visualization.jl")

# data handling (types + methods)
export Outcome, Senate, Judge, Decision, Patent
export id, label, senate, outcome, judges, date, patent, cpc, subclass, class, section
export cpc2int

# data import
export DataSource, BPatG

# plotting
export plot_posterior, errorplot!, errorplot, ridgeplot!, ridgeplot

# bayesian modeling
export transformation, sample, paramnames, predict, stats
export DynamicHMCPosterior

export BinomialGroupsModel, MixedMembershipModel, MultiMixedMembershipModel

end
