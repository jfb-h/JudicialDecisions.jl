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

# ╔═╡ ebbdca71-d2b6-47f4-9e25-90a8b5794045
using DynamicHMC: Diagnostics

# ╔═╡ 1b04c583-6c35-4f2d-b510-ebed914b60b7
using Dates

# ╔═╡ 29154953-1eb4-486b-844b-d0f52734cff2
using CairoMakie; set_theme!(theme_light())

# ╔═╡ 3b30fd71-6e74-4ebe-ad8a-a639a702a6d9
md"""
# Decision data
"""

# ╔═╡ 5c985bb7-1db0-4b3d-81bb-0bd255aeb843
decisions = JudicialDecisions.loaddata(BPatG(), "../data/json")

# ╔═╡ 8c651640-76a8-4cb6-ba63-cd749c09eaa9
filter!(decisions) do d
	Dates.year(date(d)) in 2000:2021 && length(judges(d)) > 0
end;

# ╔═╡ 61b0eb7a-6371-427f-bf93-6dad28ce91c5
md"""
# Models
"""

# ╔═╡ cecc3365-441d-4ad5-bdf5-3bafb6daf329
md"""
## Simple binomial model for variation by senate
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
problem = BinomialGroupsModel(decisions; groupfun=id ∘ senate)

# ╔═╡ f85db977-0f1e-47ce-b886-c671a792e639
post = sample(problem, 1000)

# ╔═╡ d4fcfbbe-35b2-423e-b8bf-03604b254ef7
md"""
#### Posterior inference
"""

# ╔═╡ 04f0f1b7-49db-416c-a1f3-d349401e6ebb
plot_posterior(problem, post)

# ╔═╡ ca3f7c0f-0742-44ea-82a8-26c15487cfc0
md"""
#### Key findings

"""

# ╔═╡ 892d71ca-1034-4ebc-b092-7f48aee8ffd6
md"""
The small standard deviation ($\sigma$) indicates that there is no significant variation in the probability to nullify a patent across the senates. This can also be seen by inspecting the per-senate probabilities of annullment in panel C.
"""

# ╔═╡ d87bc0ae-c45a-4e19-b654-f643a0fbf907
md"""
## Simple binomial model for variation by year
"""

# ╔═╡ 72e53e46-e133-43b4-a45e-ba128357f0c0
md"""
We can fit the some model again, however this time allowing for variation by year instead of by senates to explore statistical variation in decision outcomes over the observation period.
"""

# ╔═╡ 741c7d42-0a98-4d52-a420-574ad0efd4f4
problem_years = BinomialGroupsModel(decisions; groupfun=Dates.year ∘ date)

# ╔═╡ 6ab4c400-7d0c-4ea4-b200-7758494d1a0b
# ╠═╡ show_logs = false
post_years = sample(problem_years, 1000)

# ╔═╡ 14e315b6-2145-4f06-9267-fe291a3d6558
plot_posterior(problem_years, post_years)

# ╔═╡ e1ad7007-2adb-4bee-8079-c2c248b6618f
md"""
#### Key findings
The positive standard deviation ($\sigma$) indicates that there is some variation in the probability to nullify a patent across the 21 years in the observation period. Inspecting the per-year probabilities of annullment in panel C indicates a very slight trend towards annullment in recent years, however with strong estimation uncertainty.
"""

# ╔═╡ 2bae2ba9-25c7-40ab-953c-039d40d05308
md"""
## Variation by judge
"""

# ╔═╡ b2cb7fd6-c830-4395-8b27-dd75e2a9da86
md"""
$$\begin{align}
y_i &\sim \textrm{Bernoulli}(p_i), \text{ for } i=1,\ldots, N \\
p_i &= \alpha + \frac{1}{|J_i|} \sum_{j \in J_i} \delta_{j} \\

\alpha &\sim \textrm{Normal}(0, 1) \\
\delta_j &\sim \textrm{Normal}(0, \sigma) \\
\sigma &\sim \textrm{Exponential}(0.5)
\end{align}$$
"""

# ╔═╡ b9ac6e72-3fcf-48b0-9865-3e50afce8095
problem_judges = MixedMembershipModel(decisions)

# ╔═╡ 4a6c99ff-7ba8-437e-87a4-283d7f90cf42
# ╠═╡ show_logs = false
post_judges = sample(problem_judges, 1000; backend=:ReverseDiff)

# ╔═╡ 4d11ac83-0228-4c5b-b623-06238269b4c5
Diagnostics.summarize_tree_statistics(stats(post_judges).tree_statistics)

# ╔═╡ d43c6dd2-25bd-41af-a15c-375a4ec4127b
plot_posterior(problem_judges, post_judges, decisions; filter_predicate = >(0))

# ╔═╡ 4b3116f2-a466-4096-b48f-99b7df35fd34


# ╔═╡ 7af291e2-da0f-4dab-bd69-9fb375ab3b6f
md"""
# Setup
"""

# ╔═╡ Cell order:
# ╟─3b30fd71-6e74-4ebe-ad8a-a639a702a6d9
# ╠═5c985bb7-1db0-4b3d-81bb-0bd255aeb843
# ╠═8c651640-76a8-4cb6-ba63-cd749c09eaa9
# ╟─61b0eb7a-6371-427f-bf93-6dad28ce91c5
# ╟─cecc3365-441d-4ad5-bdf5-3bafb6daf329
# ╟─d75201a1-1885-4c72-99f7-2532177e44bb
# ╟─09efec3e-ecc8-478a-9860-340b85e773d1
# ╟─0d1dd658-3c93-44fc-ac5f-56274313c2ed
# ╠═b8dfa51a-a940-43ff-ab61-b4dd63de7cd0
# ╠═f85db977-0f1e-47ce-b886-c671a792e639
# ╟─d4fcfbbe-35b2-423e-b8bf-03604b254ef7
# ╠═04f0f1b7-49db-416c-a1f3-d349401e6ebb
# ╟─ca3f7c0f-0742-44ea-82a8-26c15487cfc0
# ╟─892d71ca-1034-4ebc-b092-7f48aee8ffd6
# ╟─d87bc0ae-c45a-4e19-b654-f643a0fbf907
# ╟─72e53e46-e133-43b4-a45e-ba128357f0c0
# ╠═741c7d42-0a98-4d52-a420-574ad0efd4f4
# ╠═6ab4c400-7d0c-4ea4-b200-7758494d1a0b
# ╠═14e315b6-2145-4f06-9267-fe291a3d6558
# ╟─e1ad7007-2adb-4bee-8079-c2c248b6618f
# ╟─2bae2ba9-25c7-40ab-953c-039d40d05308
# ╟─b2cb7fd6-c830-4395-8b27-dd75e2a9da86
# ╠═b9ac6e72-3fcf-48b0-9865-3e50afce8095
# ╠═4a6c99ff-7ba8-437e-87a4-283d7f90cf42
# ╠═ebbdca71-d2b6-47f4-9e25-90a8b5794045
# ╠═4d11ac83-0228-4c5b-b623-06238269b4c5
# ╠═d43c6dd2-25bd-41af-a15c-375a4ec4127b
# ╠═4b3116f2-a466-4096-b48f-99b7df35fd34
# ╟─7af291e2-da0f-4dab-bd69-9fb375ab3b6f
# ╠═1f7ea756-23ad-4b69-bba3-2fa15c16ae35
# ╠═23cde6f0-066a-11ed-3a4c-037299b48733
# ╠═1b04c583-6c35-4f2d-b510-ebed914b60b7
# ╠═29154953-1eb4-486b-844b-d0f52734cff2
