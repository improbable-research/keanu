package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerProxyVertex extends IntegerVertex implements ProxyVertex<IntegerVertex>, NonProbabilistic<IntegerTensor> {

    private IntegerVertex parentVertex;
    private final int[] shape;

    /**
     * This vertex acts as a "Proxy" to allow a BayesNet to be built up before parents are explicitly known (ie for
     * model in model scenarios) but allows linking at a later point in time.
     */
    public IntegerProxyVertex() {
        this(Tensor.SCALAR_SHAPE);
    }

    public IntegerProxyVertex(int[] shape) {
        this.shape = shape;
    }

    @Override
    public IntegerTensor calculate() {
        return parentVertex.getValue();
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return parentVertex.sample();
    }

    @Override
    public void setParent(IntegerVertex newParent) {
        checkTensorsMatchNonScalarShapeOrAreScalar(shape, newParent.getShape());
        parentVertex = newParent;
        setParents(parentVertex);
    }

    @Override
    public boolean hasParent() {
        return parentVertex != null;
    }

}
