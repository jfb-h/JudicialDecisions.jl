"""
    MultiMixedMembershipTimeModel

Multilevel mixed membership model for modeling the probability of annullment 
as a sum of judge effects and a sum of technology effects.

# Fields
- ys : (binary) outcome vector
- ts : year of the observation in the observation period
- js : contiguous ids of judges involved in each decision
- cs : contiguous ids of CPC technology classes of each patent
- njs : number of judges inolved in each decision
- nts : number of technologies involved in each decision
- J : Total number of judges
- T : Total number of technologies
"""
struct MultiMixedMembershipTimeModel{V,S,U,Y} <: AbstractDecisionModel
    ys::V
    ts::Y
    js::S
    cs::S
    njs::U
    ncs::U
    T::Int
    J::Int
    C::Int
end

function MultiMixedMembershipTimeModel(decisions::Vector{Decision}; levelfun=class)
    ys = (id ∘ outcome).(decisions)
    # year
    ts = Dates.year.(date.(decisions))
    ts = ts .- minimum(ts) .+ 1
    T = maximum(ts)
    # judges
    js = [id.(judges(d)) for d in decisions]
    njs = length.(js)
    J = maximum(reduce(vcat, js))
    # technologies
    cs, _ = cpc2int(decisions, levelfun)
    ncs = length.(cs)
    C = maximum(reduce(vcat, cs))
    MultiMixedMembershipTimeModel(ys, ts, js, cs, njs, ncs, T, J, C)
end

function (problem::MultiMixedMembershipTimeModel)(θ)
    @unpack α, zt, zj, zc, σt, σj, σc = θ
    @unpack ys, ts, js, cs, njs, ncs, T, J, C = problem

    loglik = sum(
        @views logpdf(Bernoulli(logistic(α + zt[t]*σt + sum(x->x*σj, zj[j]) / nj + sum(x->x*σc, zc[c]) / nt)), y) 
        for (y, t, j, c, nj, nt) in zip(ys, ts, js, cs, njs, ncs)
    )
    
    logpri = logpdf(MvNormal(T, 1), zt) +
             logpdf(MvNormal(J, 1), zj) + 
             logpdf(MvNormal(C, 1), zc) + 
             logpdf(Normal(0, 1.5), α) + 
             logpdf(Exponential(0.5), σt) + 
             logpdf(Exponential(0.5), σj) + 
             logpdf(Exponential(0.5), σc)
    
    loglik + logpri
end

function transformation(problem::MultiMixedMembershipTimeModel)
    as((
        zt=as(Array, asℝ, problem.T), 
        zj=as(Array, asℝ, problem.J), 
        zc=as(Array, asℝ, problem.C), 
        σt=asℝ₊, σj=asℝ₊, σc=asℝ₊, α=asℝ, 
    ))
end

function predict(problem::MultiMixedMembershipTimeModel, post::DynamicHMCPosterior)
    @unpack ys, ts, js, cs, njs, ncs, T, J, C = problem

    map(post) do s
        @unpack α, zt, zj, zc, σt, σj, σc = s
        map(zip(ts, js, cs, njs, ncs)) do (t, j, c, nj, nt)
            logistic(α + zt[t]*σt + sum(x->x*σj, zj[j]) / nj + sum(x->x*σc, zc[c]) / nt)
        end
    end
end
