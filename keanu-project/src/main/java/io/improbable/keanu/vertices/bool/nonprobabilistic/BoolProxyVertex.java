package io.improbable.keanu.vertices.bool.nonprobabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import com.google.common.collect.Iterables;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class BoolProxyVertex extends BoolVertex implements ProxyVertex<BoolVertex>, NonProbabilistic<BooleanTensor> {

    /**
     * This vertex acts as a "Proxy" to allow a BayesNet to be built up before parents are explicitly known (ie for
     * model in model scenarios) but allows linking at a later point in time.
     */
    public BoolProxyVertex() {
        this(Tensor.SCALAR_SHAPE);
    }

    public BoolProxyVertex(int[] shape) {
        this.setValue(BooleanTensor.placeHolder(shape));
    }

    @Override
    public BooleanTensor calculate() {
        return getParent().getValue();
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return getParent().sample();
    }

    @Override
    public void setParent(BoolVertex newParent) {
        checkTensorsMatchNonScalarShapeOrAreScalar(getShape(), newParent.getShape());
        setParents(newParent);
    }

    public BoolVertex getParent() {
        return (BoolVertex) Iterables.getOnlyElement(getParents(), null);
    }

    @Override
    public boolean hasParent() {
        return getParents().size() > 0;
    }

}
