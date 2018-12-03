package io.improbable.keanu.vertices.bool.nonprobabilistic;

import com.google.common.collect.Iterables;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class BoolProxyVertex extends BoolVertex implements ProxyVertex<BoolVertex>, NonProbabilistic<BooleanTensor> {

    private final static String LABEL_NAME = "label";

    /**
     * This vertex acts as a "Proxy" to allow a BayesNet to be built up before parents are explicitly known (ie for
     * model in model scenarios) but allows linking at a later point in time.
     *
     * @param label The label for this Vertex (all Proxy Vertices must be labelled)
     */
    public BoolProxyVertex(VertexLabel label) {
        this(Tensor.SCALAR_SHAPE, label);
    }

    public BoolProxyVertex(long[] shape, VertexLabel label) {
        super(shape);
        this.setLabel(label);
    }

    public BoolProxyVertex(@LoadVertexParam(LABEL_NAME) String label) {
        this(new VertexLabel(label));
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
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(getShape(), newParent.getShape());
        setParents(newParent);
    }

    public BoolVertex getParent() {
        return (BoolVertex) Iterables.getOnlyElement(getParents(), null);
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
