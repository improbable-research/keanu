package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import com.google.common.collect.Iterables;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import java.util.Collections;
import java.util.Map;

public class DoubleProxyVertex extends DoubleVertex
        implements ProxyVertex<DoubleVertex>, NonProbabilistic<DoubleTensor> {

    /**
     * This vertex acts as a "Proxy" to allow a BayesNet to be built up before parents are
     * explicitly known (ie for model in model scenarios) but allows linking at a later point in
     * time.
     *
     * @param label The label for this Vertex (all Proxy Vertices must be labelled)
     */
    public DoubleProxyVertex(VertexLabel label) {
        this(Tensor.SCALAR_SHAPE, label);
    }

    public DoubleProxyVertex(int[] shape, VertexLabel label) {
        this.setValue(DoubleTensor.placeHolder(shape));
        this.setLabel(label);
    }

    @Override
    public DoubleTensor calculate() {
        return getParent().getValue();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return getParent().sample();
    }

    @Override
    public void setParent(DoubleVertex newParent) {
        checkTensorsMatchNonScalarShapeOrAreScalar(getShape(), newParent.getShape());
        setParents(newParent);
    }

    public DoubleVertex getParent() {
        return (DoubleVertex) Iterables.getOnlyElement(getParents(), null);
    }

    @Override
    public boolean hasParent() {
        return !getParents().isEmpty();
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return dualNumbers.get(getParent());
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(
            PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        return Collections.singletonMap(getParent(), derivativeOfOutputsWithRespectToSelf);
    }
}
