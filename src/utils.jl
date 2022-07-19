
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
function loaddata(::BPatG, dir::String)
    files = readdir(dir)

    json = mapreduce(vcat, files) do f
        JSON3.read(read(joinpath(dir, f)), jsonlines=true)
    end

    #TODO: Better missing handling
    filter!(j -> j.outcome != "other", json)

    judgepool = mapreduce(j -> j.judges, unique ∘ vcat, json)
    judgepool = Dictionary(sort(judgepool), 1:length(judgepool))

    map(enumerate(json)) do (i, j)
        outcome = Outcome(OUTCOMES[j.outcome], j.outcome)
        senate = Senate(j.senate, "$(j.senate). Senate")
        date = Date(j.date)
        judges = map(j.judges) do j
            Judge(judgepool[j], j)
        end

        Decision(i, j.id, j.patent, outcome, date, senate, judges)
    end
end

