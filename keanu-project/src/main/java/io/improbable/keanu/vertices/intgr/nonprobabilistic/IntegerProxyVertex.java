package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import com.google.common.collect.Iterables;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class IntegerProxyVertex extends IntegerVertex implements ProxyVertex<IntegerVertex>, NonProbabilistic<IntegerTensor>, NonSaveableVertex {

    /**
     * This vertex acts as a "Proxy" to allow a BayesNet to be built up before parents are explicitly known (ie for
     * model in model scenarios) but allows linking at a later point in time.
     *
     * @param label The label for this Vertex (all Proxy Vertices must be labelled)
     */
    public IntegerProxyVertex(VertexLabel label) {
        this(Tensor.SCALAR_SHAPE, label);
    }

    public IntegerProxyVertex(long[] shape, VertexLabel label) {
        super(shape);
        setLabel(label);
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
