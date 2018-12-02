package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import com.google.common.collect.Iterables;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class IntegerProxyVertex extends IntegerVertex implements ProxyVertex<IntegerVertex>, NonProbabilistic<IntegerTensor> {

    private static final String LABEL_NAME = "label";

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

    public IntegerProxyVertex(@LoadVertexParam(LABEL_NAME) String label) {
        this(new VertexLabel(label));
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
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(getShape(), newParent.getShape());
        setParents(newParent);
    }

    public IntegerVertex getParent() {
        return (IntegerVertex) Iterables.getOnlyElement(getParents(), null);
    }

    @Override
    public boolean hasParent() {
        return !getParents().isEmpty();
    }

    @SaveVertexParam(LABEL_NAME)
    public String getLabelParameter() {
        return getLabel().toString();
    }
}
