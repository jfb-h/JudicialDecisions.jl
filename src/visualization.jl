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

function plot_posterior(problem::MixedMembershipModel, post::DynamicHMCPosterior, decisions; filter_predicate = >(0))
	fig = Figure(resolution=(800, 1200))
	
	# Hyperparameters
	color = (:grey60, .5); titlealign = :left
	
	ax1 = Axis(fig[1,1]; title="Posterior distribution of α", titlealign)
	ax2 = Axis(fig[1,2]; title="Posterior distribution of σ", titlealign)
	
	density!(ax1, post.α; color)
	density!(ax2, post.σ; color)
	
	# judge effects
	
	ax3 = Axis(fig[2,:], title="Posterior distribution of δⱼ (σzᵢ)", titlealign=:left)
	
	let
		sum = map(zip(eachrow(reduce(hcat, post.zs)), post.σ)) do (z, σ)
			(;mean=mean(z*σ), sd=std(z*σ))
		end

		idx = _filterjudges(problem, filter_predicate)

		sum = sum[idx]
		sum = sort!(sum, by= x -> getindex(x, :mean)) |> StructArray
	
		errorbars!(ax3, eachindex(sum), sum.mean, sum.sd, direction=:y, color=(:grey50, .5))
		scatter!(ax3, eachindex(sum), sum.mean, color=(:black, .5), markersize=6)
	end


	# decision probabilites

	pred = predict(problem, post)
	pred = reduce(hcat, pred)

	ax4 = Axis(fig[3,:], title="Posterior predictions for decision probabilities (pᵢ)", titlealign=:left)

	let
		ms = map(mean, eachrow(pred))
		sds =  map(std, eachrow(pred))
	
		# idx = sortperm(ms)
		# ms = ms[idx]
		# sds = sds[idx]
		# xs = eachindex(ms)
		
		xs = date.(decisions) .|> Dates.datetime2rata
		ax4.xticks = let
			y = 2000:5:2021
			d = Dates.datetime2rata.(Date.(y))
			(d, string.(y))
		end
		
		errorbars!(ax4, xs, ms, sds, direction=:y, color=(:grey50, .5))
		scatter!(ax4, xs, ms, color=(:black, .5), markersize=6)
	
		fig
	end


	 # aggregate judge probabilities

	 ax5 = Axis(fig[4,:], title="Posterior predictions for average per-judge outcome", titlealign=:left)

	 let 
		js = problem.js
		judges = reduce(vcat, js) |> unique
	
		judges = JudicialDecisions._filterjudges(problem, filter_predicate)
		
		s = map(judges) do j
			idx = findall(x -> j in x, js)
			(;mean=mean(@views pred[idx,:]), sd=std(@views pred[idx,:]))
		end
	
		s = sort!(s, by=x->getindex(x, :mean)) |> StructArray
	
		xs = eachindex(s)
	 
		errorbars!(ax5, xs, s.mean, s.sd, direction=:y, color=(:grey50, .5))
		scatter!(ax5, xs, s.mean, color=(:black, .5), markersize=6)
	
		fig
	end

	 # Labels

	 Label(fig[1,1, TopLeft()], "A", textsize=25, color=:black, padding=(0,15, 15, 0))
	 Label(fig[1,2, TopLeft()], "B", textsize=25, color=:black, padding=(0,15, 15, 0))
	 Label(fig[2,:, TopLeft()], "C", textsize=25, color=:black, padding=(0,15, 15, 0)) 
	 Label(fig[3,:, TopLeft()], "D", textsize=25, color=:black, padding=(0,15, 15, 0)) 
	 Label(fig[4,:, TopLeft()], "E", textsize=25, color=:black, padding=(0,15, 15, 0)) 

	 rowsize!(fig.layout, 1, Relative(1/6))
	 
	 fig
end