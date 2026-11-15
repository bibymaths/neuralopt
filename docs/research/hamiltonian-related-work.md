# Hamiltonian related work

This review was checked against primary papers and author/reference-manual
sources available through 15 November 2026.

Duane et al. (1987) introduced hybrid Monte Carlo; Neal's 2011 handbook chapter
established the modern practical account. Hoffman and Gelman's NUTS removes a
fixed trajectory-length choice and uses diminishing warm-up adaptation. It is
not implemented here because the first experiment isolates fixed integration
length and learned mass geometry.

Girolami and Calderhead (2011, *JRSS B*) introduced Riemannian-manifold HMC,
whose position-dependent metric requires nonseparable dynamics, log-determinant
terms, and generalized integration. NeuralOpt does not claim RMHMC: its metric
is fixed inside each transition. Betancourt's review explains why matching the
Euclidean metric to target covariance can improve exploration, but this does not
imply that an optimizer-trained preconditioner is automatically a good HMC mass.

Pakman and Paninski (2014, *JCGS*) developed exact HMC for truncated Gaussian
models; Afshar and Domke (2015, NeurIPS) studied reflection/refraction HMC.
Those methods motivate correct boundary dynamics. The present implementation
instead uses a logistic change of variables and includes its Jacobian, avoiding
claims that clipping is reversible or volume preserving.

Levy, Hoffman, and Sohl-Dickstein (2018, ICLR), “Generalizing Hamiltonian Monte
Carlo with Neural Networks,” introduced L2HMC, learning reversible
volume-preserving transformations. Hoffman et al. (2019), “NeuTra-lizing Bad
Geometry in Hamiltonian Monte Carlo,” learned transport maps before HMC. These
works learn proposal geometry while preserving explicit sampling corrections.
NeuralOpt's history-adaptive transformer metric is more modest and is treated as
an optimization heuristic outside the fixed-kernel reference experiment.

Temperature schedules relate the artificial density to simulated annealing
(Kirkpatrick, Gelatt, and Vecchi, 1983). Changing `beta` changes the target; an
annealed optimization trajectory is not a sample from one stationary density
and does not by itself prove global convergence. Temperature learning remains
unimplemented pending evidence that fixed-temperature Hamiltonian candidates
justify the added complexity.
