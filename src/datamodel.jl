
"""
    Outcome

Outcome of a judicial decision.

# Fields
- id::Int : Binary representation of the decision outcome, either 0 (not nullified) or 1 (nullified)
- label::String : Textual representation of the decision outcome
"""
struct Outcome
    id::Int
    label::String

    function Outcome(id::Int, label::String)
        id in (0, 1) || throw(ArgumentError("id needs to be 0 or 1 but was $id"))
        new(id, label)
    end
end

"""
    Senate

Senate making a judicial decision.

# Fields
- id::Int : Integer representation of the senate
- label::String : Name of the senate
"""
struct Senate
    id::Int
    label::String
end

"""
    Judge

Judge involved in making a judicial decision.

# Fields
- id::Int : Integer representation of the judge
- label::String : Name of the judge
"""
struct Judge
    id::Int
    label::String
end

"""
    Patent

Patent for which a validity decision is to be made. Contains the patent number issued 
by the respective authority (`auth` and `id`) and a vector of CPC symbols.
"""
struct Patent
    nr::String
    cpc::Vector{String}
end

Patent(s::AbstractString, c::Nothing) = Patent(s, String[])
Patent(s::AbstractString, c::AbstractArray) = Patent(s, string.(filter(!isnothing, c)))

id(x::Patent) = x.nr
cpc(x::Patent) = x.cpc
subgroup(x::Patent) = first.(x.cpc, 4) |> unique

"""
    Decision

Metadata for a judicial decisions on patent nullity.

# Fields
- id::Int : Integer id of the decision
- label::String : Official id as found on the decision document
- patent::String : id of the patent
- outcome::Outcome : decision outcome
- date::Date : date of publication of the decision
- senate::Senate : senate making the decision
- judges::Vector{Judge} : panel of judges involved in the decision
"""
struct Decision
    id::Int
    label::String
    patent::Patent
    outcome::Outcome
    date::Date
    senate::Senate
    judges::Vector{Judge}
end

id(x) = x.id
label(x) = x.label

outcome(x::Decision) = x.outcome
patent(x::Decision) = x.patent
senate(x::Decision) = x.senate
judges(x::Decision) = x.judges
date(x::Decision) = x.date

Base.show(io::IO, d::Decision) = print(io, label(d))

function Base.show(io::IO, ::MIME"text/plain", d::Decision)
    println(io, "Ruling $(label(d)) on $(id(patent(d)))")
    println(io, "Date of decision: $(Dates.format(date(d), "d U, Y"))")
    println(io, "Decided by: $(label(senate(d))) ($(join(label.(judges(d)), ", ")))")
    println(io, "Outcome: $(label(outcome(d)))")
end