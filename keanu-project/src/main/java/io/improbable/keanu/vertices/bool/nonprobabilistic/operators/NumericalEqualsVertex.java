package io.improbable.keanu.vertices.bool.nonprobabilistic.operators;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.bool.BooleanVertex;

/**
 * Returns true if a vertex value is equal to another vertex value within an epsilon.
 */
public class NumericalEqualsVertex<N extends Number, TENSOR extends NumberTensor<N, TENSOR>> extends VertexImpl<BooleanTensor> implements BooleanVertex,  NonProbabilistic<BooleanTensor> {

    protected Vertex<TENSOR> a;
    protected Vertex<TENSOR> b;
    private N epsilon;
    private final static String A_NAME = "a";
    private final static String B_NAME = "b";
    private final static String EPSILON_NAME = "epsilon";

    @ExportVertexToPythonBindings
    public NumericalEqualsVertex(@LoadVertexParam(A_NAME) Vertex<TENSOR> a,
                                 @LoadVertexParam(B_NAME) Vertex<TENSOR> b,
                                 @LoadVertexParam(EPSILON_NAME) N epsilon) {
        super(a.getShape());
        this.a = a;
        this.b = b;
        this.epsilon = epsilon;
        setParents(a, b);
    }

    @Override
    public BooleanTensor calculate() {
        return op(a.getValue(), b.getValue(), epsilon);
    }

    private BooleanTensor op(TENSOR a, TENSOR b, N epsilon) {
        return a.equalsWithinEpsilon(b, epsilon);
    }

    @SaveVertexParam(A_NAME)
    public Vertex<? extends NumberTensor> getA() {
        return a;
    }

    @SaveVertexParam(B_NAME)
    public Vertex<? extends NumberTensor> getB() {
        return b;
    }

    @SaveVertexParam(EPSILON_NAME)
    public N getEpsilon() {
        return epsilon;
    }
}
