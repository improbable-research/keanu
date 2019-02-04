### Version 0.0.19 ###

* Reorganised the factory methods for building `PosteriorSamplingAlgorithm` objects. The following are available and give you access to either a default implementation or, when you need more control over the configuration, a Builder object:
  * `Keanu.Sampling.MetropolisHastings`
  * `Keanu.Sampling.NUTS`
  * `Keanu.Sampling.SimulatedAnnealing`
  * `Keanu.Sampling.MCMC` (to automatically choose between `MetropolisHastings` and `NUTS`)
* The `PosteriorSamplingAlgorithm` objects are now immutable: you cannot set their values after construction.
* `Hamiltonian` Monte Carlo has been removed: use `NUTS` instead which is an auto-tuning implementation of Hamiltonian Monte Carlo.
* The `PosteriorSamplingAlgorithm` and `FitnessFunction` interfaces requires a `ProbabilisticModel` as its argument instead of a `BayesianNetwork`
  * You can create a `ProbabilisticModel` from a `BayesianNetwork`:
    * For `MetropolisHastings`: `new KeanuProbabilisticModel(bayesNet)`
    * For `NUTS`: `new KeanuProbabilisticModelWithGradient(bayesNet)`
* The `ProposalDistribution` interface uses `Variable` instead of `Vertex`
  * A `Variable` is an abstraction that does not assume any graph-like structure of your model.
* `KeanuRandom` has been moved to package `io.improbable.keanu`
* The `ProposalListener` interface has been changed: `onProposalApplied` is now called `onProposalCreated`.
  * This is because the `Proposal` interface no longer has `apply` and `reject` methods.
* Reversed the ordering of namespace for `VertexLabel` (e.g. from `new VertexLabel("keanu", "improbable", "io")` to `new VertexLabel("io", "improbable", "keanu"))
* Added better Python operator overloading support.
  - Includes automatically casting to the correct datatype where possible.
* Added Geometric Distribution vertex
* Fixed issue with certain vertices not taking a list properly in Kotlin
* Added `__version__` attribute to Python
* Added a permute vertex
* Added the release notes text file to the repo
