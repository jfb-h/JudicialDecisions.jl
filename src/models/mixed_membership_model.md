# Mixed Membership Model

$$
\begin{align}
y_i &\sim \textrm{Bernoulli}(p_i), \text{ for } i=1,\ldots, N \\
p_i &= \alpha + \frac{1}{|J_i|} \sum_{j \in J_i} \delta_{j} \\

\delta_j &\sim \textrm{Normal}(\mu, \sigma) \\
\mu &\sim \textrm{Normal}(0, 1) \\
\sigma &\sim \textrm{Exponential}(1)
\end{align}
$$

Where $N$ referes to the total number of decisions and $J_i$ refers to the set of judges involved in decision $i$.