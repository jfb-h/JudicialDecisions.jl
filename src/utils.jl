
abstract type DataSource end

struct BPatG end

const OUTCOMES = dictionary([
    "annulled" => 1,
    "partially annulled" => 1, 
    "claim dismissed" => 0,
    "other" => missing
])

"""
    loaddata(BPatG(), dir)

Load data in `.jsonl` format from directory dir and construct a `Vector{Decision}`.
"""
function loaddata(::BPatG, jsonfile::AbstractString)
    json = JSON3.read(read(jsonfile), jsonlines=true)

    #TODO: Better missing handling
    json = filter(json) do j
        j.outcome != "other" &&
        Date(2000) <= Date(j.date) <= Date(2021) &&
        length(j.judges) == 5 &&
        !isnothing(j.patent.cpc) &&
        length(filter(!isnothing, j.patent.cpc)) > 0
    end

    judgepool = mapreduce(j -> j.judges, unique ∘ vcat, json)
    judgepool = Dictionary(sort(judgepool), 1:length(judgepool))

    map(enumerate(json)) do (i, j)
        outcome = Outcome(OUTCOMES[j.outcome], j.outcome)
        senate = Senate(j.senate, "$(j.senate). Senate")
        patent = Patent(j.patent.nr, j.patent.cpc)
        date = Date(j.date)
        judges = map(j.judges) do j
            Judge(judgepool[j], j)
        end

        Decision(i, j.id, patent, outcome, date, senate, judges)
    end
end

 
_filterjudges(problem, predicate) = begin 
    j = reduce(vcat, problem.js)
    c = countmap(j) |> Dictionary
    filter!(predicate, c) |> keys |> collect
end

"""
    cpc2int(decisions, levelfun)

Return (1) an array of arrays containing, for each decision in `decisions`, 
a vector with the integer positions of its associated cpc symbols with respect
to (2) a sorted list of all cpc classes contained in decisions, as aggregated
via `levelfun`. 
"""
function cpc2int(decisions, levelfun)
	ts = (levelfun ∘ patent).(decisions)
    tref = sort(unique(reduce(vcat, ts)))
    ts_int = map(t -> map(i -> findfirst(==(i), tref), t), ts)
    ts_int = convert(Vector{Vector{Int}}, ts_int)
	ts_int, tref
end	

function predict_groups(pred, groups)
    p = reduce(hcat, pred)
    rows = collect(eachrow(p))
    guni = sort!(unique(reduce(vcat, groups)))
    StructArray(map(guni) do g
        i = findall(x -> g in x, groups)
        s = mean(rows[i])
        (mean=mean(s), sd=std(s))
    end)
end

function predict_groups(problem::AbstractDecisionModel, post::AbstractPosterior, groups)
    predict_groupmean(predict(problem, post), groups)
end

