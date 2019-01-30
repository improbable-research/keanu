### Version 0.0.19 ###

* Improved the structure of the Python sample dictionary
  * Scalar networks
    * If all sampled vertices are scalar then you are returned a dictionary keyed on vertex ID with the values containing a list of primitive values.
    * An example of a network of size 2 with three samples: `{"gaussian": [0., 1., 2.], "exponential": [0.5, 1.5, 3.5]}`
  * Tensor networks
    * If any of the sampled vertices are tensors then you are returned a dictionary keyed on a tuple of vertex ID and tensor index. 
    * This makes it much easier to extract all the samples for a given index and for creating a multi-index dataframe
    * An example of a scalar exponential and a `2x2` tensor gaussian with three samples:
    ```
            exp  gaussian                               
            (0)    (0, 0)    (0, 1)    (1, 0)     (1, 1)
    0  4.231334  5.017627  5.734130  3.904472   9.909033
    1  4.231334  5.017627  5.734130  3.904472   9.909033
    2  4.676046  4.308018  5.035550  6.240894  10.112683
    ```
  * As a result, whenever keying into a sample dictionary, you are guaranteed to receive a list of primitives.
  * This greatly reduces the complexity of the `autocorrelation` and `traceplot` API's as they now simply expect a list of values.  
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
