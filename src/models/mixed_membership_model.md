# Multiple Mixed Memberships Model

$$
\begin{align*}

\\
\textbf{Likelihood} \\
\\

y_{it} &\sim \textrm{Bernoulli}(p_i), \text{ for } i=1,\ldots, N \\
p_{it} &= \textrm{logit}^{-1}\Bigg(\alpha + \beta_t +
                \frac{1}{\lvert J_i \rvert} \sum_{j \in J_i} \delta_{j} +
                \frac{1}{\lvert C_i \rvert} \sum_{c \in C_i} \gamma_{c} \Bigg)
\end{align*}

\\

\begin{align*}

\\
\textbf{Priors} \\
\\

\alpha &\sim \textrm{Normal}(0, 1) \\
\beta_t &\sim \textrm{Normal}(0,\sigma_t) \\
\delta_j, \gamma_c &\sim \textrm{Normal}(0, \sigma_j), \;\; \textrm{Normal}(0, \sigma_c) \\
\sigma_t, \sigma_j, \sigma_c &\sim \textrm{Exponential}(0.5)
\end{align*}
$$
