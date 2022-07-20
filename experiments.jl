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

# ╔═╡ e20009ad-9267-4248-b9ba-8f27cd187b1a
using StatsFuns: logistic

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

# ╔═╡ d75201a1-1885-4c72-99f7-2532177e44bb
md"""
As a first simple model, we can model the probability of a patent being nullified as only depending on the senate ruling over the decision. This specification yields a convenient form utilizing the binomial distribution:

$$\begin{align}
y_s &\sim \textrm{Binomial}(n_s, p_s), \text{ for } s=1, \ldots, S \\
p_s &= \textrm{logit}^{-1}(\alpha_s) \\
\end{align}$$
"""

# ╔═╡ 09efec3e-ecc8-478a-9860-340b85e773d1
md"""
To allow for pooling of information across senates, we can utilize a hierarchical prior:

$$\begin{align}
\alpha_s &\sim \textrm{Normal}(\mu, \sigma) \\
\mu &\sim \textrm{Normal}(0, 1) \\
\sigma & \sim \textrm{Exponential}(1) \\
\end{align}$$
"""

# ╔═╡ 0d1dd658-3c93-44fc-ac5f-56274313c2ed
md"""
#### Model estimation
"""

# ╔═╡ b8dfa51a-a940-43ff-ab61-b4dd63de7cd0
problem = BinomialSenateModel(decisions)

# ╔═╡ f85db977-0f1e-47ce-b886-c671a792e639
post = sample(problem, 1000)

# ╔═╡ d4fcfbbe-35b2-423e-b8bf-03604b254ef7
md"""
#### Posterior inference
"""

# ╔═╡ 27c54416-1482-4060-af47-14d719c1a93e
mean(post.σ), std(post.σ)

# ╔═╡ 790dfd1e-0ec0-4343-ac4d-c73e677b45c6
let 
	ps = map(x -> broadcast(logistic, x), post.αs)

	x = mean(ps)
	y = 1:length(x)
	s = std(ps)

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
# ╟─d75201a1-1885-4c72-99f7-2532177e44bb
# ╟─09efec3e-ecc8-478a-9860-340b85e773d1
# ╟─0d1dd658-3c93-44fc-ac5f-56274313c2ed
# ╠═b8dfa51a-a940-43ff-ab61-b4dd63de7cd0
# ╠═f85db977-0f1e-47ce-b886-c671a792e639
# ╟─d4fcfbbe-35b2-423e-b8bf-03604b254ef7
# ╠═10086304-0529-4373-8546-9025f6d5057f
# ╠═e20009ad-9267-4248-b9ba-8f27cd187b1a
# ╠═27c54416-1482-4060-af47-14d719c1a93e
# ╟─790dfd1e-0ec0-4343-ac4d-c73e677b45c6
# ╟─7af291e2-da0f-4dab-bd69-9fb375ab3b6f
# ╠═1f7ea756-23ad-4b69-bba3-2fa15c16ae35
# ╠═23cde6f0-066a-11ed-3a4c-037299b48733
