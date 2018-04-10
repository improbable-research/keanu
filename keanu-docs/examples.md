## Examples

### Starter Project

This is a simple project that is intended to kick-start development on a new
Keanu project.

[Code](../keanu-examples/starter)

### Coal Mining Disasters

This example is a port into Keanu of the coal mining disaster example from the PYMC3 documentation (http://docs.pymc.io/notebooks/getting_started#Case-study-2:-Coal-mining-disasters).

It uses data of yearly number of coal mining accidents in the UK. The aim is to infer the "switch point" year, when the rate of accidents changes to a lower rate (this could be due to the introduction of a new safety process). The number of accidents in each year is modelled by a poisson distribution, with different rates before and after this switch point. These different rates are hyper parameters to the Poisson distribution, and are modelled as exponential vertexes.

It uses Metropolis Hastings Monte Carlo to sample the posterior distribution of the switch point, since this uses non continuous Integer Vertices you cannot use the MAP method.

[Code](../keanu-examples/coalMiningDisasters)