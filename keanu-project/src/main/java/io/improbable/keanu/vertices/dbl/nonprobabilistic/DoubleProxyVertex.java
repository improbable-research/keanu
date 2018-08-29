package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.ProxyVertex;

public class DoubleProxyVertex extends DoubleVertex implements ProxyVertex<DoubleVertex>, NonProbabilistic<DoubleTensor> {

    private DoubleVertex parentVertex;
    private final int[] shape;

    /**
     * This vertex acts as a "Proxy" to allow a BayesNet to be built up before parents are explicitly known (ie for
     * model in model scenarios) but allows linking at a later point in time.
     */
    public DoubleProxyVertex() {
        this(Tensor.SCALAR_SHAPE);
    }

    public DoubleProxyVertex(int[] shape) {
        this.shape = shape;
    }

    @Override
    public DoubleTensor calculate() {
        return parentVertex.getValue();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return parentVertex.sample();
    }

    @Override
    public void setParent(DoubleVertex newParent) {
        checkTensorsMatchNonScalarShapeOrAreScalar(shape, newParent.getShape());
        parentVertex = newParent;
        setParents(parentVertex);
    }

    @Override
    public boolean hasParent() {
        return parentVertex != null;
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return dualNumbers.get(parentVertex);
    }

}
