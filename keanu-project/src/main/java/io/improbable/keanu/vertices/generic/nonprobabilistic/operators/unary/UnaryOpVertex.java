package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;


import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.nonprobabilistic.NonProbabilistic;

public abstract class UnaryOpVertex<IN_TENSOR extends Tensor, OUT_TENSOR extends Tensor> extends NonProbabilistic<OUT_TENSOR> {

    protected final Vertex<IN_TENSOR> inputVertex;

    public UnaryOpVertex(Vertex<IN_TENSOR> inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public OUT_TENSOR sample(KeanuRandom random) {
        return op(inputVertex.sample(random));
    }

    public OUT_TENSOR getDerivedValue() {
        return op(inputVertex.getValue());
    }

    protected abstract OUT_TENSOR op(IN_TENSOR a);
}

