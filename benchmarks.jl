### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 0fb06797-5127-4dfc-aef3-7970b848f6b9
begin
	import Distributions as DS
	import MeasureTheory as MT
	using UnPack
	using Random: randexp
	using StatsFuns: logistic
	using LoopVectorization
	using DensityInterface
	import ReverseDiff
	import LogDensityProblems as LDP
	using TransformVariables, DynamicHMC
	using BenchmarkTools
end

# ╔═╡ 23df156d-0963-4041-b8cf-ba28b8fd041c
md"""
# Problem setup
"""

# ╔═╡ b6f52d3b-d7f4-405c-b980-a3f248dcbdd1
begin
	struct Problem{T,S,U}
	    ys::T
	    js::S
	    ns::U
	    J::Int
	end

	function Problem(N, J, params)
		@unpack α, zs, σ = params
		ns = rand(3:5, N)
		js = [rand(1:J, n) for n in ns]
		ys = map(zip(js, ns)) do (j, n)
			logistic(α + sum(x -> x*σ, @views zs[j]) / n)
		end
		ys = rand.(DS.Bernoulli.(ys))
		Problem(ys, js, ns, J)
	end

	#(problem::Problem)(params) = logjoint_man(problem, params)
end

# ╔═╡ 7cb56fbc-d6a8-4e4a-aae6-7db8f71ac0cb
function randparams(J)
    zs = randn(J)
    α = randn()
    σ = randexp()
    (; α, zs, σ)
end

# ╔═╡ ceeed582-eef9-4242-8ff1-4940835d7f9c
N, J = 1000, 200

# ╔═╡ 3b8fca81-7933-4b28-8d48-a64ea063a292
params = randparams(J);

# ╔═╡ dcfa96c5-2ea7-44a8-b26e-6f1c578d5bba
problem = Problem(N, J, params);

# ╔═╡ 8ce091b2-ee48-4ffa-bc33-ee844b21bda0
md"""
# Logdensity functions
"""

# ╔═╡ 63da3302-f7f5-4159-b9ed-25b7f1c31665
d = MT.For{MT.Normal{(:μ, :σ), Tuple{Float64, Float64}}}([1.0,2.0,3.0]) do x 
	MT.Normal(x, 1) 
end

# ╔═╡ d94083e5-794b-4e88-b150-97e7a006a9fe
r = rand(d); typeof(r)

# ╔═╡ 64751d13-7310-41a7-85af-174f9a9908e7
@btime logdensityof($d, $r)

# ╔═╡ 563219ad-cdc0-4b11-befe-c8f93cf70092
function logjoint_dists(problem, θ)
    @unpack α, zs, σ = θ
    @unpack ys, js, ns, J = problem

    loglik = sum(
		logdensityof(DS.Bernoulli(logistic(α + @views sum(x -> x*σ, zs[j]) / n)), y) 
		for (y, j, n) in zip(ys, js, ns)
	)
	
    logpri = logdensityof(DS.MvNormal(J, 1.0), zs) + 
			 logdensityof(DS.Normal(0, 1.5), α) + 
			 logdensityof(DS.Exponential(0.5), σ)
    
	loglik + logpri
end

# ╔═╡ afc4a5af-4edc-4553-aefc-48d886f794b2
function logjoint_mt(problem, θ)
    @unpack α, zs, σ = θ
    @unpack ys, js, ns, J = problem
	
	d = MT.For(js, ns) do j, n
		@views MT.Bernoulli(logistic(α + sum(x -> x*σ, zs[j]) / n))
	end
	
    loglik = logdensityof(d, ys)
	
    logpri = logdensityof(MT.Normal() ^ J, zs) + 
			 logdensityof(MT.Normal(0, 1.5), α) + 
			 logdensityof(MT.Exponential(0.5), σ)
    
	loglik + logpri
end

# ╔═╡ 24bc4320-cf2a-4f9d-ba95-85bcb0feb3dd
# ╠═╡ disabled = true
#=╠═╡
function logjoint_man(problem, θ)
    @unpack α, zs, σ = θ
    @unpack ys, js, ns, J = problem
	
    loglik = 0.0

	# for (y, j, n) in zip(ys, js, ns)
	# 	@views p = logistic(α + sum(x -> x*σ, zs[j]) / n)
	# 	loglik += logdensityof(MT.Bernoulli(p), y)
	# end

	for (i, y) in enumerate(ys)
		p = @views logistic(α + sum(x->x*σ, zs[js[i]]) / ns[i])
		loglik += logdensityof(MT.Bernoulli(p), y)
	end

	# loglik = ThreadsX.mapreduce(+, ys, js, ns; init=0.0) do y, j, n
	# 	@views p = logistic(α + sum(x -> x*σ, zs[j]) / n)
	#  	logdensityof(MT.Bernoulli(p), y)
	# end
	
    logpri = logdensityof(MT.Normal() ^ J, zs) + 
			 logdensityof(MT.Normal(0, 1.5), α) + 
			 logdensityof(MT.Exponential(0.5), σ)
    
	loglik + logpri
end
  ╠═╡ =#

# ╔═╡ 03c4e4e1-1e37-4901-a3a5-f64dc00486d4
md"""
# Logdensity benchmarks
"""

# ╔═╡ a268ef82-de80-4c2c-88f2-a1125255176e
rc = true # toggle for running benchmarks

# ╔═╡ af72de05-f9f5-4b1b-ac7a-d902a416a7b2
rc; @btime logjoint_dists($problem, $params)

# ╔═╡ fa933928-291a-4a92-802c-9fe2ea716dca
rc; @btime logjoint_mt($problem, $params)

# ╔═╡ 73b67ae2-0d49-4834-bed6-524a3e7eedf8
#=╠═╡
rc; @btime logjoint_man($problem, $params)
  ╠═╡ =#

# ╔═╡ 0fdc3409-d65c-48c5-84cf-cacba7c4d195
md"""
# Gradient benchmarks
"""

# ╔═╡ 0dc02c81-3b66-4637-8859-4387fd46104d
t = as((zs=as(Array, asℝ, maximum(reduce(vcat, problem.js))), α=asℝ, σ=asℝ₊))

# ╔═╡ 8b438fe1-9e33-47ec-a116-e23e6d4091f3
md"""
### ReverseDiff
"""

# ╔═╡ d35fa4d6-99a9-4e05-9d43-603af9725d28
rc; begin
	ℓ1 = LDP.TransformedLogDensity(t, par->logjoint_dists(problem, par))
	∇ℓ1 = LDP.ADgradient(:ReverseDiff, ℓ1; compile=Val(true))
	@btime LDP.logdensity_and_gradient($∇ℓ1, $zeros(LDP.dimension(ℓ1)))
end

# ╔═╡ 70cec591-9a4c-4bd3-9479-52d34c7124eb
rc; begin
	ℓ2 = LDP.TransformedLogDensity(t, par->logjoint_mt(problem, par))
	∇ℓ2 = LDP.ADgradient(:ReverseDiff, ℓ2; compile=Val(true))
	@btime LDP.logdensity_and_gradient($∇ℓ2, $zeros(LDP.dimension(ℓ2)))
end

# ╔═╡ 70d3229b-d084-4ecd-9598-95b442184e6b
#=╠═╡
rc; begin
	ℓ3 = LDP.TransformedLogDensity(t, par->logjoint_man(problem, par))
	∇ℓ3 = LDP.ADgradient(:ReverseDiff, ℓ3; compile=Val(true))
	@btime LDP.logdensity_and_gradient($∇ℓ3, $zeros(LDP.dimension(ℓ3)))
end
  ╠═╡ =#

# ╔═╡ 4d60f003-13fd-4641-9bb9-fe1261034df7
md"""
### ForwardDiff
"""

# ╔═╡ 1461f26e-0fba-4bc3-8df0-8490a022ddcd
rc; begin
	ℓ4 = LDP.TransformedLogDensity(t, par->logjoint_dists(problem, par))
	∇ℓ4 = LDP.ADgradient(:ForwardDiff, ℓ4)
	@btime LDP.logdensity_and_gradient($∇ℓ4, $zeros(LDP.dimension(ℓ4)))
end

# ╔═╡ 82e718f7-4b02-4a7a-8ee0-6a0d49a189f5
rc; begin
	ℓ5 = LDP.TransformedLogDensity(t, par->logjoint_mt(problem, par))
	∇ℓ5 = LDP.ADgradient(:ForwardDiff, ℓ5)
	@btime LDP.logdensity_and_gradient($∇ℓ5, $zeros(LDP.dimension(ℓ5)))
end

# ╔═╡ 4aedf85b-c649-4f92-858e-a86357224fba
#=╠═╡
rc; begin
	ℓ6 = LDP.TransformedLogDensity(t, par->logjoint_man(problem, par))
	∇ℓ6 = LDP.ADgradient(:ForwardDiff, ℓ6)
	@btime LDP.logdensity_and_gradient($∇ℓ6, $zeros(LDP.dimension(ℓ6)))
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
DynamicHMC = "bbc10e6e-7c05-544b-b16e-64fede858acb"
LogDensityProblems = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890"
MeasureTheory = "eadaa1a4-d27c-401d-8699-e962e1bbc33b"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
TransformVariables = "84d833dd-6860-57f9-a1a7-6da5db126cff"
UnPack = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"

[compat]
BenchmarkTools = "~1.3.1"
DensityInterface = "~0.4.0"
Distributions = "~0.25.65"
DynamicHMC = "~3.1.2"
LogDensityProblems = "~0.11.5"
LoopVectorization = "~0.12.118"
MeasureTheory = "~0.16.5"
ReverseDiff = "~1.14.1"
StatsFuns = "~1.0.1"
TransformVariables = "~0.6.2"
UnPack = "~1.0.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0-rc1"
manifest_format = "2.0"
project_hash = "7040cff37b95b21df378440a9ac1fa8e2252c0a1"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "c877a35749324754d3c8fffb09fc1f9db144ff8f"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.18"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["ArrayInterfaceCore", "Compat", "IfElse", "LinearAlgebra", "Static"]
git-tree-sha1 = "d956c0606a3bc1112a1f99a8b2309b79558d9921"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "6.0.17"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "7d255eb1d2e409335835dc8624c35d97453011eb"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.14"

[[deps.ArrayInterfaceOffsetArrays]]
deps = ["ArrayInterface", "OffsetArrays", "Static"]
git-tree-sha1 = "7dce0e2846e7496622f5d2742502d7e029693458"
uuid = "015c0d05-e682-4f19-8f0a-679ce4c54826"
version = "0.1.5"

[[deps.ArrayInterfaceStaticArrays]]
deps = ["Adapt", "ArrayInterface", "LinearAlgebra", "Static", "StaticArrays"]
git-tree-sha1 = "d7dc30474e73173a990eca86af76cae8790fa9f2"
uuid = "b0d46f97-bff5-4637-a19a-dd75974142cd"
version = "0.1.2"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "ebe4bbfc4de38ef88323f67d60a4e848fb550f0e"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "0.8.9"

[[deps.ArraysOfArrays]]
deps = ["Adapt", "ChainRulesCore", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "33177b879bb757a900b035eb8b49b4e8af938572"
uuid = "65a8f2f4-9b39-5baf-92e2-a9cc46fdf018"
version = "0.6.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "eaee37f76339077f86679787a71990c4e465477f"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.4"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "Static"]
git-tree-sha1 = "8a43595f7b3f7d6dd1e07ad9b94081e1975df4af"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.1.25"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CloseOpenIntervals]]
deps = ["ArrayInterface", "Static"]
git-tree-sha1 = "5522c338564580adf5d58d91e43a55db0fa5fb39"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "59d00b3139a9de4eb961057eabb65ac6522be954"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.0"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "429077fd74119f5ac495857fd51f4120baf36355"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.65"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "c5544d8abb854e306b7b2f799ab31cdba527ccae"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicHMC]]
deps = ["ArgCheck", "DocStringExtensions", "LinearAlgebra", "LogDensityProblems", "Parameters", "ProgressMeter", "Random", "Statistics"]
git-tree-sha1 = "e3001279bd373f904278da0e7d06d9a9b7b187ec"
uuid = "bbc10e6e-7c05-544b-b16e-64fede858acb"
version = "3.1.2"

[[deps.DynamicIterators]]
deps = ["Random", "Trajectories"]
git-tree-sha1 = "089b6dc3f3c4d651142724386fd37b508f30e4d4"
uuid = "6c76993d-992e-5bf1-9e63-34920a5a5a38"
version = "0.4.2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GeneralizedGenerated]]
deps = ["DataStructures", "JuliaVariables", "MLStyle", "Serialization"]
git-tree-sha1 = "60f1fa1696129205873c41763e7d0920ac7d6f1f"
uuid = "6b9d7cbe-bcb9-11e9-073f-15a7a543e2eb"
version = "0.3.3"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "b7b88a4716ac33fe31d6556c02fc60017594343c"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.8"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Infinities]]
git-tree-sha1 = "b2732e2076cd50639d827f9ae9fc4ea913c927fe"
uuid = "e1ba4f0e-776d-440f-acd9-e1d2e9742647"
version = "0.1.4"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KeywordCalls]]
deps = ["Compat", "Tricks"]
git-tree-sha1 = "f1ebacfe730add8513798fbd833d360112780033"
uuid = "4d827475-d3e4-43d6-abe3-9688362ede9f"
version = "0.2.4"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "ArrayInterfaceOffsetArrays", "ArrayInterfaceStaticArrays", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static"]
git-tree-sha1 = "b67e749fb35530979839e7b4b606a97105fe4f1c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.10"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "MatrixFactorizations", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "d9a962fac652cc6b0224622b18199f0ed46d316a"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "0.22.11"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.81.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DiffResults", "DocStringExtensions", "Random", "Requires", "TransformVariables", "UnPack"]
git-tree-sha1 = "f81b04e2c50b8e781e6171aee9ff7cc99f239abd"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "0.11.5"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7c88f63f9f0eb5929f15695af9a4d7d3ed278a91"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.16"

[[deps.LogarithmicNumbers]]
deps = ["Random"]
git-tree-sha1 = "4833b079cfa1cf2960cfb3d05e500bba841ff26a"
uuid = "aa2f6b4e-9042-5d33-9679-40d3a6b85899"
version = "1.2.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "ArrayInterfaceOffsetArrays", "ArrayInterfaceStaticArrays", "CPUSummary", "ChainRulesCore", "CloseOpenIntervals", "DocStringExtensions", "ForwardDiff", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "SIMDDualNumbers", "SIMDTypes", "SLEEFPirates", "SpecialFunctions", "Static", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "5ea9a0aaf5ded7f0b6e43c96ca1793e60c96af93"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.118"

[[deps.MLStyle]]
git-tree-sha1 = "c4f433356372cc8838da59e3608be4b0c4c2c280"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.13"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "2212d36f97e01347adb1460a6914e20f2feee853"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "0.9.1"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.MeasureBase]]
deps = ["ChainRulesCore", "ChangesOfVariables", "Compat", "ConstructionBase", "DensityInterface", "FillArrays", "IfElse", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "LogarithmicNumbers", "MappedArrays", "NaNMath", "PrettyPrinting", "Random", "Reexport", "Static", "Statistics", "Test", "Tricks"]
git-tree-sha1 = "4fa8c7505106f85bedeca29380cfc706dedf1650"
uuid = "fa1605e6-acd5-459c-a1e6-7e635759db14"
version = "0.12.2"

[[deps.MeasureTheory]]
deps = ["Accessors", "ChangesOfVariables", "Compat", "ConcreteStructs", "ConstructionBase", "DensityInterface", "Distributions", "DynamicIterators", "FillArrays", "IfElse", "Infinities", "InteractiveUtils", "InverseFunctions", "KeywordCalls", "LazyArrays", "LinearAlgebra", "LogExpFunctions", "MLStyle", "MacroTools", "MappedArrays", "MeasureBase", "NamedTupleTools", "NestedTuples", "PositiveFactorizations", "PrettyPrinting", "Random", "Reexport", "SpecialFunctions", "Static", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "TransformVariables", "Tricks"]
git-tree-sha1 = "f22fb61f6a0726b14c2a1d6de80e9e4b22f076f8"
uuid = "eadaa1a4-d27c-401d-8699-e962e1bbc33b"
version = "0.16.5"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NamedTupleTools]]
git-tree-sha1 = "befc30261949849408ac945a1ebb9fa5ec5e1fd5"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.14.0"

[[deps.NestedTuples]]
deps = ["Accessors", "ArraysOfArrays", "BangBang", "GeneralizedGenerated", "NamedTupleTools", "Static"]
git-tree-sha1 = "3f9217cee8728997e46b541da3599b20bf9dc19b"
uuid = "a734d2a7-8d68-409b-9419-626914d4061d"
version = "0.3.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "cf82af4e114b0da31c4896aef6c5b8be3fe0916d"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.1.7"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyPrinting]]
git-tree-sha1 = "4be53d093e9e37772cc89e1009e8f6ad10c4681b"
uuid = "54e16d92-306c-5ea0-a30b-337be88ac337"
version = "0.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.ReverseDiff]]
deps = ["ChainRulesCore", "DiffResults", "DiffRules", "ForwardDiff", "FunctionWrappers", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "SpecialFunctions", "StaticArrays", "Statistics"]
git-tree-sha1 = "b8e2eb3d8e1530acb73d8949eab3cedb1d43f840"
uuid = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
version = "1.14.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDDualNumbers]]
deps = ["ForwardDiff", "IfElse", "SLEEFPirates", "VectorizationBase"]
git-tree-sha1 = "dd4195d308df24f33fb10dde7c22103ba88887fa"
uuid = "3cdde19b-5bb0-4aaf-8931-af3e248e098b"
version = "0.1.1"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "7ee0e13ac7cd77f2c0e93bff8c40c45f05c77a5a"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.33"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "5d2c08cef80c7a3a8ba9ca023031a85c263012c5"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "23368a3313d12a2326ad0035f0db0c0966f438ef"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "66fe9eb253f910fe8cf161953880cfdaef01cdf0"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.1"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2c11d7290036fe7aac9038ff312d3b3a2a5bf89e"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.4.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "472d044a1c8df2b062b23f222573ad6837a615ba"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.19"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "f8629df51cab659d70d2e5618a430b4d3f37f2c3"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.0"

[[deps.Trajectories]]
deps = ["RecipesBase", "Tables"]
git-tree-sha1 = "9c7a662752d8b5dd43afd56384738590a58a4cdc"
uuid = "2c80a279-213e-54d7-a557-e9a14725db56"
version = "0.2.2"

[[deps.TransformVariables]]
deps = ["ArgCheck", "ChangesOfVariables", "DocStringExtensions", "ForwardDiff", "InverseFunctions", "LinearAlgebra", "LogExpFunctions", "Pkg", "Random", "UnPack"]
git-tree-sha1 = "0bee5465eb2dff94a964794f6b2a2438e41fb770"
uuid = "84d833dd-6860-57f9-a1a7-6da5db126cff"
version = "0.6.2"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static"]
git-tree-sha1 = "81d19dae338dd4cf3ecd6331fb4763a1002f9580"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.43"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.41.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═0fb06797-5127-4dfc-aef3-7970b848f6b9
# ╟─23df156d-0963-4041-b8cf-ba28b8fd041c
# ╠═b6f52d3b-d7f4-405c-b980-a3f248dcbdd1
# ╠═7cb56fbc-d6a8-4e4a-aae6-7db8f71ac0cb
# ╠═ceeed582-eef9-4242-8ff1-4940835d7f9c
# ╠═3b8fca81-7933-4b28-8d48-a64ea063a292
# ╠═dcfa96c5-2ea7-44a8-b26e-6f1c578d5bba
# ╟─8ce091b2-ee48-4ffa-bc33-ee844b21bda0
# ╠═63da3302-f7f5-4159-b9ed-25b7f1c31665
# ╠═d94083e5-794b-4e88-b150-97e7a006a9fe
# ╠═64751d13-7310-41a7-85af-174f9a9908e7
# ╠═563219ad-cdc0-4b11-befe-c8f93cf70092
# ╠═afc4a5af-4edc-4553-aefc-48d886f794b2
# ╠═24bc4320-cf2a-4f9d-ba95-85bcb0feb3dd
# ╟─03c4e4e1-1e37-4901-a3a5-f64dc00486d4
# ╠═a268ef82-de80-4c2c-88f2-a1125255176e
# ╠═af72de05-f9f5-4b1b-ac7a-d902a416a7b2
# ╠═fa933928-291a-4a92-802c-9fe2ea716dca
# ╠═73b67ae2-0d49-4834-bed6-524a3e7eedf8
# ╟─0fdc3409-d65c-48c5-84cf-cacba7c4d195
# ╠═0dc02c81-3b66-4637-8859-4387fd46104d
# ╟─8b438fe1-9e33-47ec-a116-e23e6d4091f3
# ╠═d35fa4d6-99a9-4e05-9d43-603af9725d28
# ╠═70cec591-9a4c-4bd3-9479-52d34c7124eb
# ╠═70d3229b-d084-4ecd-9598-95b442184e6b
# ╟─4d60f003-13fd-4641-9bb9-fe1261034df7
# ╠═1461f26e-0fba-4bc3-8df0-8490a022ddcd
# ╠═82e718f7-4b02-4a7a-8ee0-6a0d49a189f5
# ╠═4aedf85b-c649-4f92-858e-a86357224fba
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
