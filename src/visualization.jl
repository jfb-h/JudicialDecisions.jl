function plot_posterior(problem::BinomialGroupsModel, post::DynamicHMCPosterior)
	fig = Figure(resolution=(800, 900))
	
	color = (:grey60, .5); titlealign = :left
	
	ax1 = Axis(fig[1,1]; title="Posterior distribution of logistic(μ)", titlealign)
	ax2 = Axis(fig[2,1]; title="Posterior distribution of σ", titlealign)
	
	density!(ax1, logistic.(post.μ); color)
	density!(ax2, post.σ; color)

	m = sum(problem.ys) / sum(problem.ns)
	ps = map(x -> broadcast(logistic, x), post.αs)
	x = mean(ps); y = 1:length(x)
	s = std(ps)
	
	xticks = round.(sort(vcat(m, 0, .5, 1)), digits=2)
	
	ax3 = Axis(fig[3,1],
		yticks=(y, string.(problem.group)), 
		xticks= (xticks, string.(xticks)),
		xlabel="Probability of (partial) annullment", 
		title="Posterior distribution of αs",
		titlealign=:left,
	)

	vlines!(ax3, m, color=:grey70)
	errorbars!(ax3, x, y, 2s, direction=:x)
	scatter!(ax3, x, y, color=:black)
	xlims!(ax3, 0, 1)

	Label(fig[1,1, TopLeft()], "A", textsize=25, color=:black, padding=(0,15, 15, 0))
	Label(fig[2,1, TopLeft()], "B", textsize=25, color=:black, padding=(0,15, 15, 0))
	Label(fig[3,1, TopLeft()], "C", textsize=25, color=:black, padding=(0,15, 15, 0))

	rowsize!(fig.layout, 3, Relative(.6))
	
	fig
end

function plot_posterior(problem::MixedMembershipModel, post::DynamicHMCPosterior; filter_predicate = >(0))
	fig = Figure(resolution=(800, 800))
	
	# Hyperparameters
	color = (:grey60, .5); titlealign = :left
	
	ax1 = Axis(fig[1,1]; title="Posterior distribution of α", titlealign)
	ax2 = Axis(fig[1,2]; title="Posterior distribution of σ", titlealign)
	
	density!(ax1, post.α; color)
	density!(ax2, post.σ; color)
	
	# per judge probabilities
	sum = map(eachrow(reduce(hcat, post.δs))) do r
		(;mean=mean(r), sd=std(r))
	end
 
	 filterjudges(problem, predicate) = begin 
		 j = reduce(vcat, problem.js)
		 c = countmap(j) |> Dictionary
		 filter!(predicate, c) |> keys |> collect
	 end

	 idx = filterjudges(problem, filter_predicate)

 
	 sum = sum[idx]
	 sum = sort!(sum, by= x -> getindex(x, :mean)) |> StructArray
 
 
	 ax2 = Axis(fig[2,:])
 
	 errorbars!(ax2, eachindex(sum), sum.mean, sum.sd, direction=:y)
	 scatter!(ax2, eachindex(sum), sum.mean, color=:black)

	 rowsize!(fig.layout, 2, Relative(.7))
	 
	 fig
end