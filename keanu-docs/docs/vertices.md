## Vertices

There are several families of vertices. Each family shares a common value type.

### Probabilistic

Probabilistic vertices are vertices that do not depend completely on their parent vertices. An example
of this is a vertex that acts as a probability distribution like the GaussianVertex.

Changing the value of their parent vertices may change the density (from pdf) at their value but it 
will not change the value of the vertex.

### Non-Probabilistic

The value of these vertices are completely dependent on their parent vertices values. For example,
given A * B = C, C is a non-probabilistic vertex. A or B might be probabilistic vertices, which would
make C not a constant value but still completely dependent on A and B.

### The Double Family

A DoubleVertex can be used by most arithmetic operators. They can be used to describe a problem
that can be solved using gradient ascent optimization.

* Probabilistic
    * Beta
    * Exponential
    * Gamma
    * Gaussian
    * Laplace
    * Logistic
    * Smooth Uniform
    * Triangular
    * Uniform
    
* Non-Probabilistic
    * Operators
    
### The Integer Family

An IntegerVertex can also be used by most arithmetic operators.

* Probabilistic
    * Poisson
    * Uniform
    * Fuzzy Cast To Integer
    
* Non-Probabilistic
    * Operators

### The Boolean (true/false) Family

A BoolVertex can be used by most boolean operators. These can be observed directly and used in MCMC.

* Probabilistic
    * Flip
    
* Non-Probabilistic
    * Operators
    * Boolean Cast

### The Generic family

These are the vertices that can have any type as a value. This type can be an Enum or a user defined object.
An example of this is the SelectVertex<T> where T is any type.

```java
    public enum MyType {
        A, B, C, D
    }

    public SelectVertex<MyType> getSelectorForMyType() {

        LinkedHashMap<MyType, DoubleVertex> frequency = new LinkedHashMap<>();
        frequency.put(A, ConstantVertex.of(0.25));
        frequency.put(B, ConstantVertex.of(0.25));
        frequency.put(C, ConstantVertex.of(0.25));
        frequency.put(D, ConstantVertex.of(0.25));

        return new SelectVertex<MyType>(frequency);
    }
```

The getSelectorForMyType() method would return a probabilistic vertex that would contain an 
object of type MyType A, B, C or D, 25% of the time respectively.

* Probabilistic
    * Select

* Non-Probabilistic
    * Operators
    * Constant
    * If
    * Multiplexer