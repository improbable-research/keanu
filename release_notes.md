### Version 0.0.19 ###

* Added Adam optimizer
* GradientOptimizer and NonGradientOptimizer now takes an algorithm argument that by default will use Conjugate Gradient and BOBYQA respectively.
* Reorganised the factory methods for building `PosteriorSamplingAlgorithm` objects. The following are available and give you access to either a default implementation or, when you need more control over the configuration, a Builder object:
  * `Keanu.Sampling.MetropolisHastings`
  * `Keanu.Sampling.NUTS`
  * `Keanu.Sampling.SimulatedAnnealing`
  * `Keanu.Sampling.MCMC` (to automatically choose between `MetropolisHastings` and `NUTS`)
* The `PosteriorSamplingAlgorithm` objects are now immutable: you cannot set their values after construction.
* When you choose a custom configuration of `MetropolisHastings` or `SimulatedAnnealing`, you must specify:
  * the proposal distribution - the default option has been removed. Options are `PriorProposalDistribution` and `GaussianProposalDistribution`
  * a proposal rejection strategy - options are `RollBackToCachedValuesOnRejection` and `RollbackAndCascadeOnRejection`
    * (in Python, this is done for you: you only have to pass in the set of latent vertices)
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
* Added `__version__` attribute to Python
* Added a permute vertex
* Added the release notes text file to the repo
