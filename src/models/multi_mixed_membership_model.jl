"""
    MultiMixedMembershipModel

Multilevel mixed membership model for modeling the probability of annullment as a sum of judge effects and a sum of technology effects.

# Fields
- ys : (binary) outcome vector
- js : contiguous ids of judges involved in each decision
- ts : contiguous ids of technologies involved in each decision
- njs : number of judges inolved in each decision
- nts : number of technologies involved in each decision
- J : Total number of judges
- T : Total number of technologies
"""
struct MultiMixedMembershipModel{T,S,U} <: AbstractDecisionModel
    ys::T
    js::S
    ts::S
    njs::U
    nts::U
    J::Int
    T::Int
end

function MultiMixedMembershipModel(decisions::Vector{Decision}; levelfun=class)
    ys = (id ∘ outcome).(decisions)
    # judges
    js = [id.(judges(d)) for d in decisions]
    njs = length.(js)
    J = maximum(reduce(vcat, js))
    # technologies
    ts, _ = cpc2int(decisions, levelfun)
    nts = length.(ts)
    T = maximum(reduce(vcat, ts))
    MultiMixedMembershipModel(ys, js, ts, njs, nts, J, T)
end

function (problem::MultiMixedMembershipModel)(θ)
    @unpack α, zj, zt, σj, σt = θ
    @unpack ys, js, ts, njs, nts, J, T = problem

    loglik = sum(
        @views logpdf(Bernoulli(logistic(α + sum(x->x*σj, zj[j]) / nj + sum(x->x*σt, zt[t]) / nt)), y) 
        for (y, j, t, nj, nt) in zip(ys, js, ts, njs, nts)
    )
    
    logpri = logpdf(MvNormal(J, 1), zj) + 
             logpdf(MvNormal(T, 1), zt) + 
             logpdf(Normal(0, 1.5), α) + 
             logpdf(Exponential(0.5), σj) + 
             logpdf(Exponential(0.5), σt)
    
    loglik + logpri
end

function transformation(problem::MultiMixedMembershipModel)
    as((
        zj=as(Array, asℝ, maximum(reduce(vcat, problem.js))), 
        zt=as(Array, asℝ, maximum(reduce(vcat, problem.ts))), 
        σj=asℝ₊, σt=asℝ₊,α=asℝ, 
    ))
end

function predict(problem::MultiMixedMembershipModel, post::DynamicHMCPosterior)
    @unpack ys, js, ts, njs, nts, J, T = problem

    map(post) do s
        @unpack α, zj, zt, σj, σt = s
        map(zip(js, ts, njs, nts)) do (j, t, nj, nt)
            logistic(α + sum(x->x*σj, zj[j]) / nj + sum(x->x*σt, zt[t]) / nt)
        end
    end
end
