## Version 0.0.24

## Version 0.0.23

### Common

* You can now build a `Sequence` using more than one factory, whose vertices may be dependent on another.
  * In Java: you can call `SequenceBuilder#withFactories` instead of `SequenceBuilder#withFactory`
  * In Python: the parameter has been renamed from `factory` to `factories` (BREAKING CHANGE). You can pass in one or a list of multiple factories. 

## Version 0.0.22

### Common

* `DoubleVertex#matrixMultiply` now performs dot product when given two vectors, and matrix-vector product when given a matrix and a vector.
* Autodiff for Slice Vertices has now been fixed (caused issues with NUTs when run on a graph containing Slices)
* `MultivariateGaussianVertex` now requires that mu is a vector instead of a matrix.
* `DoubleTensor.scalar` will throw an exception if the DoubleTensor is not actually a scalar. Please use .getValue(0) if you just
want the first element in a DoubleTensor.

### Python

* Improved performance of getting samples by coercing them into a `Tensor` in Java (instead of iterating through an `ArrayList`).

## Version 0.0.21 

### Common

* Using Buildkite for CI
* Can save dot files with disconnected vertices in BayesianNetwork. You can also just pass a list of vertices to DotSaver.
* Plates have been renamed
  * This was done primarily because we were using the term incorrectly, since Dynamic Bayes Nets (e.g. Hidden Markov) are not Plates.
  * `Plates` --> `Sequence`
  * `Plate` --> `SequenceItem`
  * `plate` package --> `template`
* `ForwardSampler`
  * Samples from the prior of a bayesian network. The network cannot have observations that are dependent on random variables.
  * It's accessible from the `Keanu` factory like so: `Keanu.Sampling.Forward.withDefaultConfig()`
  * There is a togglabe option to record the log prob of each sample. This is disabled by default for performance reasons.
  
### Python

* Expose `unobserve`
* Added `iter_all_vertices`
* Rename methods returning generators to `iter_*` from `get_*`
* Split the ND4J dependencies into their own pip package.

## Version 0.0.20

### Common

* Saving a network as a DOT file includes labels on constant vertices.

### Python

* Improved performance of getting samples by using byte streams.
* Added Python docstrings for sampling

## Version 0.0.19

* Added `get_vertex_by_label` to `BayesNet`
* Added optional param `label` for all vertices in Python (e.g. `Gaussian(0., 1., label="gaussian")`). Now you must label vertices you are sampling from, otherwise Keanu will throw an exception.
* Improved the structure of the Python sample dictionary
  * Scalar networks
    * If all sampled vertices are scalar then you are returned a dictionary keyed on vertex ID with the values containing a list of primitive values.
    * An example of a network of size 2 with three samples: `{"gaussian": [0., 1., 2.], "exponential": [0.5, 1.5, 3.5]}`
    * `samples["gaussian"] = [0., 1., 2.]`
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
    * `samples[("gaussian", (0, 1))] = [5.734130, 5.734130, 5.035550] `
  * As a result, whenever keying into a sample dictionary, you are guaranteed to receive a list of primitives.
  * This greatly reduces the complexity of the `autocorrelation` and `traceplot` API's as they now simply expect a list of values.  
* Added Adam optimizer
* GradientOptimizer and NonGradientOptimizer now takes an algorithm argument that by default will use Conjugate Gradient and BOBYQA respectively.
* GradientOptimizer and NonGradientOptimizer return a OptimizedResult object instead of just the optimized fitness as a double
* Reorganised the factory methods for building `PosteriorSamplingAlgorithm` objects. The following are available and give you access to either a default implementation or, when you need more control over the configuration, a Builder object:
  * `Keanu.Sampling.MetropolisHastings`
  * `Keanu.Sampling.NUTS`
  * `Keanu.Sampling.SimulatedAnnealing`
  * `Keanu.Sampling.MCMC` (to automatically choose between `MetropolisHastings` and `NUTS`)
* The `PosteriorSamplingAlgorithm` objects are now immutable: you cannot set their values after construction.
* When you choose a custom configuration of `MetropolisHastings` or `SimulatedAnnealing`, you can specify:
  * the proposal distribution. Options are `PriorProposalDistribution` and `GaussianProposalDistribution`
  * a proposal rejection strategy - options are `RollBackToCachedValuesOnRejection` and `RollbackAndCascadeOnRejection`
    * (in Python, this is done for you)
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
* Added the MIRSaver/MIRLoader + the proposed MIR proto format
* Added the release notes text file to the repo
