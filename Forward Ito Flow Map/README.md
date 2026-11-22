# Strong Forward Flow Maps for a Multiplicative SDE: The CIR Process

This repository contains a standalone Jupyter notebook  that learns the **strong forward flow map** (the forward Itô map) of a Cox–Ingersoll–Ross (CIR) process from simulated data and evaluates the learned map on held-out paths.

The map takes only the initial state $X_0$ and the Legendre coefficients $\hat{I}$ of the driving Brownian path on each interval.


## Motivation

*Strong Stochastic Flow Maps* (McCallum, Blasingame et al., 2026) learns a map  $\Psi(x,\hat{I},h)$  that sends a state $X_s$ and the Legendre coefficients $\hat{I}$ of the Brownian path on $[s, s+h]$ to the state $X_{s+h}$ on the same path. Training combines a tangent loss on short windows with a semigroup (self-distillation) loss that composes the map over two half-windows.

The original formulation uses an additive parameterization, $y + hf + wg$ with $w = W_{s+h} - W_s$, whose existence argument relies on additive noise. In contrast, the CIR process

$$
\mathrm{d}X_t = \theta(\mu - X_t)\,\mathrm{d}t + \sigma\sqrt{X_t}\,\mathrm{d}W_t
$$

has multiplicative noise whose diffusion coefficient is not Lipschitz at $0$. Under the Feller condition $2\theta\mu \ge \sigma^2$, the process remains strictly positive. An additive map has no structural guarantee of preserving positivity: a single step can produce a negative value, causing subsequent steps to leave the domain of the process.

This notebook tests whether the SSFM framework extends to multiplicative SDEs by constructing a flow map that is **strictly positive by construction**.

### Strong Flow for CIR

Applying the Lamperti transform $Y_t = \sqrt{X_t}$ yields

$$
\mathrm{d}Y_t = \left(\frac{\alpha}{Y_t} - \frac{\theta}{2}Y_t\right)\mathrm{d}t + \frac{\sigma}{2}\,\mathrm{d}W_t, \qquad \alpha = \frac{4\theta\mu - \sigma^2}{8} > 0,
$$

which has additive noise and a smooth drift on $(0, \infty)$. Under the Feller condition, $Y_t$ stays in $(0, \infty)$, so $X_t = Y_t^2$ is a well-defined function of $X_s$ and the Brownian increments on $[s, t]$. This defines the forward Itô map approximated by the network. By the comparison theorem, the flow is strictly increasing in $x$, and its Jacobian is random with expectation $e^{-\theta h}$.

### Positive-Preserving Simulation Data

Training and evaluation paths are simulated using the drift-implicit Euler scheme on $\sqrt{X_t}$ (Alfonsi, 2005; strong convergence proved in Dereich, Neuenkirch, and Szpruch, 2012). Each time step solves a quadratic equation whose two roots have product

$$
-\frac{\alpha\,\Delta t}{1 + \frac{\theta}{2}\Delta t} < 0,
$$

guaranteeing a unique positive root for every Brownian increment.

* **Parameters:** $\theta = 1.5$, $\mu = 0.5$, $\sigma = 0.8$ (Feller ratio $2\theta\mu/\sigma^2 = 2.34$)
* **Initial distribution & horizon:** $X_0 \sim \mathrm{Gamma}(4, 0.3)$, $T = 2.5$, $1200$ fine steps
* **Positivity check:** Across all $24\text{ million}$ simulated states, the minimum value is $4.9 \cdot 10^{-4}$.

### Multiplicative Flow Architecture

The additive update is replaced by an exponential parameterization:

$$
\Psi_\nu(x, \hat{I}, h) = x\,e^{E}, \qquad E = c\tanh(\tilde{E}/c), \qquad \tilde{E} = h\,M^{(1)}_\nu(z, \hat{I}, \sqrt{h}) + w\,M^{(2)}_\nu(z, \hat{I}, \sqrt{h}),
$$

with $c = 10$ and saturated log-state input $z = 8\tanh(\log(x/\mu)/8)$.

* **Itô structure:** Over a short window, $\log(X_t/X_s)$ decomposes into a drift term proportional to $h$ and a diffusion term proportional to $w$, which are modeled by the two network heads $M^{(1)}$ and $M^{(2)}$.
* **Positivity & stability:** The exponential guarantees $\Psi_\nu > 0$ for all weights. The $\tanh$ cap restricts single-step ratios to $e^{\pm 10}$, while the saturated input $z$ prevents wild extrapolation below the training range.
* **Asymptotic identification:** Expanding the exponential as $h \to 0$ recovers the drift and diffusion coefficients directly from the heads:

$$
x\left(M^{(1)} + \frac{1}{2}\left(M^{(2)}\right)^2\right) \to b(x), \qquad \sqrt{x}\,M^{(2)} \to \sigma.
$$

### Training Setup

* **Objective:** Joint SSFM loss ($75\%$ tangent loss, $25\%$ semigroup loss with an EMA teacher and stop-gradient).
* **Data-only supervision:** Tangent targets are observed states over windows of $1$–$4$ fine cells, with residuals weighted by $1/(h\,x_s)$. Neither the true drift nor diffusion enters the loss; both are recovered purely from trajectory data.
* **Tail oversampling:** $25\%$ of each tangent batch is sampled from states below the $5\%$ quantile (since only $0.5\%$ of raw training states fall below $0.05$).
* **Optimization:** All gradients are implemented analytically in NumPy and verified against finite differences. The model is trained for $12{,}000$ iterations with batch size $4096$, frozen, and evaluated on $4000$ held-out test paths.


## Main Results (Test Split, Single Run)

All reported errors are root-mean-square errors (RMSE) normalized by the exact standard deviation of the target state, making the identity map $\Psi(x) = x$ the baseline reference. Diagnostic figures are available inside the notebook.

### Single Call on One Window

| Window $h$ | $0.0042$ | $0.0167$ | $0.05$ | $0.1$ | $0.167$ | $0.5$ | $1.0$ | $2.5$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Learned map** | $0.0010$ | $0.0053$ | $0.0168$ | $0.0314$ | $0.0459$ | $0.0848$ | $0.114$ | $0.137$ |
| **Identity baseline** | $0.102$ | $0.206$ | $0.355$ | $0.494$ | $0.642$ | $1.076$ | $1.517$ | $2.808$ |

### Chained Reconstruction of $X_T$ from $X_0$ ($n$ Steps)

The strong error averages across $n \in \{20, 24, 25, 30, 40, 50\}$:

| Steps $n$ | $1$ | $2$ | $5$ | $10$ | $20$ | $50$ | Strong Error ($95\%$ Bootstrap CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test RMSE** | $0.138$ | $0.136$ | $0.102$ | $0.086$ | $0.079$ | $0.075$ | $0.0771$ ($0.0748,\ 0.0799$) |

* **Validation strong error:** $0.0789$
* **Per-path test strong error ($4000$ paths):** Median $0.058$, $99\%$ quantile $0.208$, maximum $0.758$.

### Distribution of Path Minima ($n = 50$, $4000$ Test Paths)

| Minimum Interval | Truth | Learned Map |
| :--- | :---: | :---: |
| $(-\infty, 0)$ | $0$ | $0$ |
| $\{0\}$ | $0$ | $0$ |
| $(0, 10^{-6})$ | $0$ | $1$ |
| $[10^{-6}, 10^{-4})$ | $0$ | $1$ |
| $[10^{-4}, 10^{-3})$ | $2$ | $6$ |
| $[10^{-3}, 0.01)$ | $24$ | $25$ |
| $[0.01, 0.02)$ | $81$ | $40$ |
| $[0.02, 0.05)$ | $391$ | $214$ |
| $[0.05, 0.1)$ | $784$ | $614$ |
| $[0.1, \infty)$ | $2718$ | $3099$ |

### Recovered Drift & Diffusion Coefficients

Drift and diffusion coefficients are extracted from the frozen map via Monte Carlo integration over $\hat{I} \sim \mathcal{N}(0, I)$ on windows of length $h$, then fitted to $\theta(\mu - x)$ and $\sigma\sqrt{x}$. Richardson extrapolation removes the $\mathcal{O}(h)$ finite-window bias.

| Estimator | $\hat{\theta}$ | $\hat{\mu}$ | $\hat{\sigma}$ | $\hat{\theta}$ (Richardson) | $\hat{\mu}$ (Richardson) | $\hat{\sigma}$ (Richardson) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Map ($h = 0.01$)** | $1.498$ | $0.481$ | $0.806$ | $1.508$ | $0.482$ | $0.853$ |
| **Map ($h = 0.02$)** | $1.488$ | $0.479$ | $0.757$ | $1.513$ | $0.474$ | $0.785$ |
| **Ground truth** | $1.500$ | $0.500$ | $0.800$ | $1.500$ | $0.500$ | $0.800$ |


## Limitations

* **Undershoots near zero:** At $n = 50$, $8$ test paths reach a reconstructed minimum below $10^{-3}$ (compared to $2$ in the ground truth), with a global minimum node of $8.0 \cdot 10^{-7}$ versus $7.7 \cdot 10^{-4}$ for the true paths. While strictly positive, a small fraction of trajectories dip closer to zero than the true process.
* **Slight positive bias:** The signed error on $X_T$ is positive on $70\%$ of test paths (mean $+0.02$ standard deviations; relative mean $+5\%$), corresponding to fewer learned paths having minima in $[0.01, 0.05)$.
* **Loss of monotonicity on very long windows:** For single calls at $h = 1.0$ and $h = 2.5$, $\min \partial_x \Psi < 0$, meaning single long-horizon calls are not globally monotone in $x$ (and the Jacobian cocycle residual reaches $1.9$ at $h = 2.5$). Chained evaluations with $n \ge 5$ use windows of length $h \le 0.5$, where $\partial_x \Psi > 0$ holds everywhere.
* **Extrapolation cap below the training range:** For states below the observed data support, the map's median still pushes states upward, but more weakly than the exact transition law. Because the exact median from a state near $0$ approaches a positive constant, the required log-ratio grows like $\log(1/x)$ and exceeds the $c = 10$ cap below $x \approx 10^{-5}$.
* **Marginal distribution fit:** The empirical law of reconstructed $X_T$ ($n = 50$) closely tracks the exact stationary/transition distribution (Kolmogorov–Smirnov distance $0.023$), though a two-sample KS test rejects equality at the $5\%$ level ($p = 0.03$).



## References

* S. McCallum, Z. W. Blasingame et al. (2026). *Strong Stochastic Flow Maps*.
* A. Alfonsi (2005). On the discretization schemes for the CIR (and Bessel squared) processes. *Monte Carlo Methods and Applications*, 11(4), 355–384.
* S. Dereich, A. Neuenkirch, and L. Szpruch (2012). An Euler-type method for the strong approximation of the Cox–Ingersoll–Ross process. *Proceedings of the Royal Society A*, 468(2140), 1105–1115.
