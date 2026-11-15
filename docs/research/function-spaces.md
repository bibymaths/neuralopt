# Periodic function-space reference experiment

There is no infinite-dimensional Lebesgue measure. Function-space probability
is formulated relative to a Gaussian reference measure, following Beskos et al.
(2011) and Cotter et al. (2013). NeuralOpt's initial periodic reference uses the
spectral covariance

\[C_k=(\alpha+|k|^2)^{-p},\qquad p>1/2,\]

and real Fourier synthesis. The compact reconstruction problem minimizes a
mesh-normalized L2 data fit plus periodic H1 regularization. Its Fourier solution
is analytic, providing an independent refinement reference.

Tests at 32, 64, 128, and 256 points recover the same continuous sine/cosine
solution and verify a near-zero discrete gradient. This establishes spectral
operator correctness and discretization consistency only. It does **not** show
that the transformer transfers across resolutions, that a learned operator is
mesh independent, or that the finite discretization converges as a probability
measure in function space.

Dense learned matrices are deliberately absent. A future learned geometry must
act relative to the reference covariance through Fourier multipliers or
resolution-stable low-rank corrections.
