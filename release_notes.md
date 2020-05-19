## Version 0.0.27

### Java

#### Breaking changes
- Moved `io.improbable.keanu.vertices.dbl` classes to `io.improbable.keanu.vertices.tensor.number.floating.dbl`
- Moved `io.improbable.keanu.vertices.int` classes to `io.improbable.keanu.vertices.tensor.number.fixed.int`
- Moved `io.improbable.keanu.vertices.bool` classes to `io.improbable.keanu.vertices.tensor.bool`
- renamed DoubleTensor `determinant()` method `matrixDeterminant()`
- renamed DoubleTensor `average()` to `mean()` and allowed it to be done on a given dimension

### Common

- Previously some operations were available on the tensor classes but not the vertex classes. All operations
on the tensor classes have been implemented with vertex operations now. A few operations have been added in order
to accommodate autodiff for the new vertex operations.

- All Tensor operations now have a corresponding Vertex. This brings the DoubleTensor, IntegerTensor and BooleanTensor
  in line with the DoubleVertex, IntegerVertex and BooleanVertex.   

- Along with this change, the vertices package has been changed:
io.improbable.keanu.vertices.* -> io.improbable.keanu.vertices.tensor.*

- There is now an experimental LongTensor implementation.
- There is now a native JVM implementation of the IntegerTensor. This will bring a significant performance increase 
to models with lots of small integer operations.

#### New operations

- sign
- strided slice
- trigamma
- triLower
- triUpper
- fillTriangular
- batch matrix multiply
- batch matrix inverse
- batch matrix determinant
- batch cholesky decomposition
- cholesky inverse (with batch)


## Version 0.0.26

### Common

* Made constructing the children set of a vertex more efficient by not constantly reconstructing the set.
* Upgrade to nd4j 1.0.0-beta4

### Java

* Fixed an error where a space was present in one of the dependencies, which broke some build systems fetching dependencies.

- Fixed issues with MultinomialVertex. The issues with it before were:
    - It was extremely strict on checking that p summed to 1.0
    - It disallowed scalar n. This is probably the most common use case.
    - It didn't support the usual batch sampling nor batch logProb (see MultinomialVertex.java for docs)

- MultinomialVertex Functional changes:
    - The shape of p expected k to be the far left dimension. It is now the far right in order to allow for broadcasting semantics.
    - n and p parameter validation was semi-controllable but is now completely toggleable with `vertex.setValidationEnabled(true);`

- Add more generic tensor slice that allows start stop and interval slicing.

- Change name on DoubleTensor compare methods by removing get (e.g. `getGreaterThanMask(...))` -> `greaterThanMask(...)`)

- BooleanTensor now uses boolean arrays instead of Nd4j doubles

- Added custom distribution example to examples

- Added to DoubleTensor operations inspired by numpy: 
    - cumSum
    - cumProd
    - product
    - logAddExp
    - logAddExp2
    - log10
    - log2
    - log1p
    - exp2
    - expM1
    - tanh
    - atanh
    - sinh
    - asinh
    - cosh
    - acosh

- Fixed bug in NUTS where NaN gradients do not cause a step to be divergent

### Python

* Faster tensor creation in Python


## Version 0.0.25

## Version 0.0.24

### Common

* Sequences save / load

## Version 0.0.23

### Common

* You can now build a `Sequence` using more than one factory, whose vertices may be dependent on another.
  * In Java: you can call `SequenceBuilder#withFactories` instead of `SequenceBuilder#withFactory`
  * In Python: the parameter has been renamed from `factory` to `factories` (BREAKING CHANGE). You can pass in one or a list of multiple factories. 

## Version 0.0.22

### Common

* Generalised `GaussianProposalDistribution` to take different sigmas for each latent.
  * Example in Python:
    * `MetropolisHastingsSampler(proposal_distribution='gaussian', latents=[v1, v2], proposal_distribution_sigma=[1., np.array([[1., 2.], [3., 4.]]))`
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
