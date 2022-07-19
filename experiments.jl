### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 1f7ea756-23ad-4b69-bba3-2fa15c16ae35
# ╠═╡ show_logs = false
using Pkg; Pkg.activate(".")

# ╔═╡ 23cde6f0-066a-11ed-3a4c-037299b48733
# ╠═╡ show_logs = false
begin
	using Revise
	using JudicialDecisions
end

# ╔═╡ 10086304-0529-4373-8546-9025f6d5057f
using CairoMakie

# ╔═╡ 84959732-8563-4758-b92e-9ccaaa042098
md"""
# TODO

- [x] Datamodel
- [x] Simple senate model

"""

# ╔═╡ 3b30fd71-6e74-4ebe-ad8a-a639a702a6d9
md"""
# Decision data
"""

# ╔═╡ 5c985bb7-1db0-4b3d-81bb-0bd255aeb843
decisions = JudicialDecisions.loaddata(BPatG(), "../data/json")

# ╔═╡ 485bb1e9-3632-4b09-9b80-212d19fa080e
decisions[15]

# ╔═╡ 61b0eb7a-6371-427f-bf93-6dad28ce91c5
md"""
# Models
"""

# ╔═╡ cecc3365-441d-4ad5-bdf5-3bafb6daf329
md"""
## Simple binomial model for senates
"""

# ╔═╡ 09efec3e-ecc8-478a-9860-340b85e773d1
md"""
$$y_s \sim \textrm{Binomial}(n_s, p_s), \;\; \mbox{for} \; i=1, \ldots, S$$
"""

# ╔═╡ b8dfa51a-a940-43ff-ab61-b4dd63de7cd0
problem = SenateModelPosterior(decisions)

# ╔═╡ f85db977-0f1e-47ce-b886-c671a792e639
post = sample(problem, 1000)

# ╔═╡ 43eb2fdd-4b38-45d8-94e2-76d11e848f58
length(post), paramnames(post)

# ╔═╡ 6978de71-2b89-49df-a6d7-a4a30a6dd6cc
mean(post.ps)

# ╔═╡ 790dfd1e-0ec0-4343-ac4d-c73e677b45c6
let m = mean(post.ps), s = std(post.ps)

	x = m; y = 1:length(m)
	xmin = m .- 2s
	xmax = m .+ 2s

	axis = (;yticks=(y, string.(y)), ylabel="Senate")
	figure = (;resolution=(800, 400))
	errorbars(x, y, 2s, direction=:x; axis, figure)
	scatter!(x, y, color=:black)
	xlims!(0, 1)
	current_figure()
end


# ╔═╡ 7af291e2-da0f-4dab-bd69-9fb375ab3b6f
md"""
# Setup
"""

# ╔═╡ Cell order:
# ╟─84959732-8563-4758-b92e-9ccaaa042098
# ╟─3b30fd71-6e74-4ebe-ad8a-a639a702a6d9
# ╠═5c985bb7-1db0-4b3d-81bb-0bd255aeb843
# ╠═485bb1e9-3632-4b09-9b80-212d19fa080e
# ╟─61b0eb7a-6371-427f-bf93-6dad28ce91c5
# ╟─cecc3365-441d-4ad5-bdf5-3bafb6daf329
# ╠═09efec3e-ecc8-478a-9860-340b85e773d1
# ╠═b8dfa51a-a940-43ff-ab61-b4dd63de7cd0
# ╠═f85db977-0f1e-47ce-b886-c671a792e639
# ╠═43eb2fdd-4b38-45d8-94e2-76d11e848f58
# ╠═6978de71-2b89-49df-a6d7-a4a30a6dd6cc
# ╠═10086304-0529-4373-8546-9025f6d5057f
# ╠═790dfd1e-0ec0-4343-ac4d-c73e677b45c6
# ╟─7af291e2-da0f-4dab-bd69-9fb375ab3b6f
# ╠═1f7ea756-23ad-4b69-bba3-2fa15c16ae35
# ╠═23cde6f0-066a-11ed-3a4c-037299b48733
