# Generalization

Coordinate inputs and gradients are explicitly transformed into normalized box
coordinates. If `u = 2(x-l)/(h-l)-1`, then `df/du = (h-l) df/dx / 2`.
Round-trip behavior is regression tested.

The transformer and the new shared-coordinate MLP baseline have parameter
counts independent of objective dimension. The MLP consumes the same coordinate
features plus budget and dimension signals, but has no attention; it is smaller
than the configured transformer and provides an executable architecture
ablation.

Only one unseen-dimension smoke result is currently executed: the 12-update
regret model worsened an unseen 8D rotated, condition-100 quadratic (164.71
versus 89.41 for v0.1.0). There is not yet enough evidence for interpolation or
extrapolation claims. Unseen-family and multi-seed studies are outstanding.
