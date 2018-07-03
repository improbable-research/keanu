## Overview

### Describing your problem

You describe your problem as a directed acyclic graph using vertices to either deterministically define computations on downstream values or to probabilistically state your belief of a value at a point in the network. This is called a Bayesian Network.

## Your Bayesian Network is a Directed Acyclic Graph

Your model needs to be described to Keanu as a Bayesian network. This network contains
vertices and your model's state (i.e. data) is housed in these vertices as the vertex's `value`. 
The value of a vertex can depend on the value of a parent vertex and can be updated in one of
two ways.

Given the example of two vertices A, B that contain some number. The number from A and B are multiplied together,
which yields C.

```
(A) _
     \
      * -> (C)
(B) _/
```

If the number in A changes then the number in C should change as well and likewise for changes from B.

### Propagating changes forward

If you change A, you can tell C to recalculate based off of A's new value and B's unchanged value. 
This can be done by: 

```
A.setAndCascade(1.234)
```

### Evaluating upstream changes

But if you want to change both A and B then you probably don't want to have C update twice. In that
case you would prefer to calculate C after both and A and B have changed and therefore calculating 
C once. This can be done by:

```
A.setValue(1.234)
B.setValue(4.321)
C.lazyEval()
```

### Observing a value

Another central concept to Bayesian networks is observations. The value of a vertex can be "observed", which
effectively locks the value of the vertex. Observing a vertex raises a flag on the vertex that tells an
inference algorithm to treat the vertex as fixed on that value.

Observing vertices that contain numbers is a special case and is described more in the docs on [Double vertices](vertices.md).
In the interest to keeping this simple, take for example where instead of multiplying A and B, we AND their values.

```
(A) _
     \
     AND -> (C)
(B) _/
```

In this example, A and B contain either true or false and C is true if both A AND B are true. If you were to
observe that C is true like:

```
C.observe(true)
```

then you can infer that A and B are also both true.

### Latent (i.e. hidden)

If a vertex is probabilistic but not observed then it is considered a latent vertex. The value of these
vertices can be inferred using an inference algorithm.

