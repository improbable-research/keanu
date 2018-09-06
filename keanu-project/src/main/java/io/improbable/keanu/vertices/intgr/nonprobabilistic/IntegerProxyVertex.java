package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import com.google.common.collect.Iterables;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerProxyVertex extends IntegerVertex implements ProxyVertex<IntegerVertex>, NonProbabilistic<IntegerTensor> {

    /**
     * This vertex acts as a "Proxy" to allow a BayesNet to be built up before parents are explicitly known (ie for
     * model in model scenarios) but allows linking at a later point in time.
     */
    public IntegerProxyVertex() {
        this(Tensor.SCALAR_SHAPE);
    }

    public IntegerProxyVertex(int[] shape) {
        setValue(IntegerTensor.placeHolder(shape));
    }

    public IntegerProxyVertex(String label) {
        this();
        setLabel(new VertexLabel(label));
    }

    @Override
    public IntegerTensor calculate() {
        return getParent().getValue();
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return getParent().sample();
    }

    @Override
    public void setParent(IntegerVertex newParent) {
        checkTensorsMatchNonScalarShapeOrAreScalar(getShape(), newParent.getShape());
        setParents(newParent);
    }

    public IntegerVertex getParent() {
        return (IntegerVertex) Iterables.getOnlyElement(getParents(), null);
    }

    @Override
    public boolean hasParent() {
        return !getParents().isEmpty();
    }

}
