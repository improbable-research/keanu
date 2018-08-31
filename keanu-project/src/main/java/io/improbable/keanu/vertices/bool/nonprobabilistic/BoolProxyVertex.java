package io.improbable.keanu.vertices.bool.nonprobabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class BoolProxyVertex extends BoolVertex implements ProxyVertex<BoolVertex>, NonProbabilistic<BooleanTensor> {

    private BoolVertex parentVertex;
    private final int[] shape;

    /**
     * This vertex acts as a "Proxy" to allow a BayesNet to be built up before parents are explicitly known (ie for
     * model in model scenarios) but allows linking at a later point in time.
     */
    public BoolProxyVertex() {
        this(Tensor.SCALAR_SHAPE);
    }

    public BoolProxyVertex(int[] shape) {
        this.shape = shape;
    }

    @Override
    public BooleanTensor calculate() {
        return parentVertex.getValue();
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return parentVertex.sample();
    }

    @Override
    public void setParent(BoolVertex newParent) {
        checkTensorsMatchNonScalarShapeOrAreScalar(shape, newParent.getShape());
        parentVertex = newParent;
        setParents(parentVertex);
    }

    @Override
    public boolean hasParent() {
        return parentVertex != null;
    }

}
