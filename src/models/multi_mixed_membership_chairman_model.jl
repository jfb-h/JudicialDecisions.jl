"""
    MultiMixedMembershipChairmanModel

Multilevel mixed membership model for modeling the probability of annullment 
as a sum of judge effects and a sum of technology effects. Also includes variation over time 
with a per-year random effect and allows for the chairman's influence to be higher than the 
other judges'.

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
struct MultiMixedMembershipChairmanModel{S} <: AbstractDecisionModel
    ys::S
    ts::Vector{Int}
    js::Vector{Vector{Int}}
    cs::Vector{Vector{Int}}
    njs::Vector{Int}
    ncs::Vector{Int}
    T::Int
    J::Int
    C::Int
end

function MultiMixedMembershipChairmanModel(decisions::Vector{Decision}; levelfun=class)
    ys = (id ∘ outcome).(decisions)
    # year
    ts = Dates.year.(date.(decisions))
    ts = ts .- minimum(ts) .+ 1
    T = maximum(ts)
    # judges (first judge is taken to be the chairman)
    js = [id.(judges(d)) for d in decisions]
    njs = length.(js)
    J = maximum(reduce(vcat, js))
    # technologies
    cs, _ = cpc2int(decisions, levelfun)
    ncs = length.(cs)
    C = maximum(reduce(vcat, cs))
    MultiMixedMembershipChairmanModel(ys, ts, js, cs, njs, ncs, T, J, C)
end

w(ζ, n) = [ζ, fill((1-ζ) / (n-1), n-1)...]

function (problem::MultiMixedMembershipChairmanModel)(θ)
    @unpack α, ζ, zt, zj, zc, σt, σj, σc = θ
    @unpack ys, ts, js, cs, njs, ncs, T, J, C = problem

    loglik = sum(
        @views logpdf(Bernoulli(logistic(α + zt[t]*σt + w(ζ, nj)'*(zj[j]*σj) + sum(x->x*σc, zc[c]) / nc)), y) 
        for (y, t, j, c, nj, nc) in zip(ys, ts, js, cs, njs, ncs)
    )
    
    logpri = logpdf(MvNormal(T, 1), zt) +
             logpdf(MvNormal(J, 1), zj) + 
             logpdf(MvNormal(C, 1), zc) + 
             logpdf(Normal(0, 1.5), α) + 
             logpdf(Beta(2, 2), ζ) + 
             logpdf(Exponential(0.5), σt) + 
             logpdf(Exponential(0.5), σj) + 
             logpdf(Exponential(0.5), σc)
    
    loglik + logpri
end

function transformation(problem::MultiMixedMembershipChairmanModel)
    as((
        zt=as(Array, asℝ, problem.T), 
        zj=as(Array, asℝ, problem.J), 
        zc=as(Array, asℝ, problem.C), 
        σt=asℝ₊, σj=asℝ₊, σc=asℝ₊, 
        α=asℝ, ζ=as𝕀,
    ))
end

function predict(problem::MultiMixedMembershipChairmanModel, post::DynamicHMCPosterior)
    @unpack ys, ts, js, cs, njs, ncs, T, J, C = problem

    map(post) do s
        @unpack α, ζ, zt, zj, zc, σt, σj, σc = s
        map(zip(ts, js, cs, njs, ncs)) do (t, j, c, nj, nc)
            @views logistic(α + zt[t]*σt + w(ζ, nj)'*(zj[j]*σj) + sum(x->x*σc, zc[c]) / nc)
        end
    end
end
