# Multiple Mixed Memberships Model

$$
\begin{align*}

\\
\textbf{Likelihood} \\
\\

y_i &\sim \textrm{Bernoulli}(p_i), \text{ for } i=1,\ldots, N \\
p_i &= \textrm{logit}^{-1}\Bigg(\alpha + \frac{1}{|J_i|} \sum_{j \in J_i} \delta_{j} +
                \frac{1}{|C_i|} \sum_{c \in C_i} \gamma_{c} \Bigg) \\

\\
\textbf{Priors} \\
\\

\alpha &\sim \textrm{Normal}(0, 1) \\
\delta_j, \gamma_c &\sim \textrm{Normal}(0, \sigma_j), \;\; \textrm{Normal}(0, \sigma_c) \\
\sigma_j, \sigma_c &\sim \textrm{Exponential}(0.5)
\end{align*}
$$
