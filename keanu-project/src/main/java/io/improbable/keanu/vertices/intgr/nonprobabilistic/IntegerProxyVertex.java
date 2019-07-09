package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import com.google.common.collect.Iterables;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class IntegerProxyVertex extends VertexImpl<IntegerTensor> implements IntegerVertex, ProxyVertex<IntegerVertex>, NonProbabilistic<IntegerTensor> {

    private static final String LABEL_NAME = "label";
    private static final String PARENT_NAME = "parent";

    /**
     * This vertex acts as a "Proxy" to allow a BayesNet to be built up before parents are explicitly known (ie for
     * model in model scenarios) but allows linking at a later point in time.
     *
     * @param label The label for this Vertex (all Proxy Vertices must be labelled)
     */
    public IntegerProxyVertex(VertexLabel label) {
        this(Tensor.SCALAR_SHAPE, label);
    }

    @ExportVertexToPythonBindings
    public IntegerProxyVertex(long[] shape, VertexLabel label) {
        super(shape);
        setLabel(label);
    }

    public IntegerProxyVertex(@LoadShape long[] shape, @LoadVertexParam(LABEL_NAME) String labelString, @LoadVertexParam(value = PARENT_NAME, isNullable = true) IntegerVertex parent) {
        super(shape);
        VertexLabel vertexLabel = VertexLabel.parseLabel(labelString);
        setLabel(vertexLabel);
        if (parent != null) {
            setParent(parent);
        }
    }

    @Override
    public <V extends Vertex<IntegerTensor>> V setLabel(VertexLabel label) {
        if (this.getLabel() != null && !this.getLabel().getUnqualifiedName().equals(label.getUnqualifiedName())) {
            throw new RuntimeException("You should not change the label on a Proxy Vertex");
        }
        return super.setLabel(label);
    }

    public IntegerProxyVertex(long[] tensorShape, String label) {
        this(tensorShape, new VertexLabel(label));
    }

    @Override
    public IntegerTensor calculate() {
        return getParent().getValue();
    }

    @Override
    public void setParent(IntegerVertex newParent) {
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(getShape(), newParent.getShape());
        setParents(newParent);
    }

    @SaveVertexParam(value = PARENT_NAME, isNullable = true)
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
