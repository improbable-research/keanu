package io.improbable.keanu.vertices.bool.nonprobabilistic.operators;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;

/**
 * Returns true if a vertex value is equal to another vertex value within an epsilon.
 */
public class NumericalEqualsVertex extends BooleanVertex implements NonProbabilistic<BooleanTensor> {

    protected Vertex<? extends NumberTensor> a;
    protected Vertex<? extends NumberTensor> b;
    private Vertex<? extends NumberTensor> epsilon;
    private final static String A_NAME = "a";
    private final static String B_NAME = "b";
    private final static String EPISILON_NAME = "episilon";

    @ExportVertexToPythonBindings
    public NumericalEqualsVertex(@LoadVertexParam(A_NAME) Vertex<? extends NumberTensor> a,
                                 @LoadVertexParam(B_NAME) Vertex<? extends NumberTensor> b,
                                 @LoadVertexParam(EPISILON_NAME) Vertex<? extends NumberTensor> epsilon) {
        super(a.getShape());
        this.a = a;
        this.b = b;
        this.epsilon = epsilon;
        setParents(a, b, epsilon);
    }

    @Override
    public BooleanTensor calculate() {
        return op(a.getValue(), b.getValue(), epsilon.getValue());
    }

    private BooleanTensor op(NumberTensor a, NumberTensor b, NumberTensor epsilon) {
        DoubleTensor absoluteDifference = a.toDouble().minus(b.toDouble()).absInPlace();
        return absoluteDifference.lessThanOrEqual(epsilon.toDouble());
    }

    @SaveVertexParam(A_NAME)
    public Vertex<? extends NumberTensor> getA() {
        return a;
    }

    @SaveVertexParam(B_NAME)
    public Vertex<? extends NumberTensor> getB() {
        return b;
    }

    @SaveVertexParam(EPISILON_NAME)
    public Vertex<? extends NumberTensor> getEpsilon() {
        return epsilon;
    }
}
