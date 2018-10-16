package io.improbable.keanu.model.regression;

import java.util.function.Function;

import io.improbable.keanu.model.ModelGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import lombok.Getter;
import lombok.Value;

public class LinearRegressionGraph<OUTPUT> implements ModelGraph<DoubleTensor, OUTPUT> {
    @Getter
    private final DoubleVertex xVertex;
    @Getter
    private final Vertex<OUTPUT> yVertex;
    @Getter
    private final Vertex<OUTPUT> yObservationVertex;
    @Getter
    private final DoubleVertex weightsVertex;
    @Getter
    private final DoubleVertex interceptVertex;
    @Getter
    private final BayesianNetwork net;

    public LinearRegressionGraph(long[] featureShape, Function<DoubleVertex, OutputVertices<OUTPUT>> outputTransform, DoubleVertex interceptVertex, DoubleVertex weightsVertex) {
        long featureCount = featureShape[0];
        TensorShapeValidation.checkShapesMatch(weightsVertex.getShape(), new long[]{1, featureCount});
        TensorShapeValidation.checkShapesMatch(interceptVertex.getShape(), new long[]{1, 1});
        this.weightsVertex = weightsVertex;
        this.interceptVertex = interceptVertex;
        xVertex = new ConstantDoubleVertex(DoubleTensor.zeros(featureShape));
        OutputVertices<OUTPUT> outputVertices = outputTransform.apply(
            TensorShape.isScalar(weightsVertex.getShape()) ?
                weightsVertex.times(xVertex).plus(interceptVertex) :
                weightsVertex.matrixMultiply(xVertex).plus(interceptVertex)
        );
        yVertex = outputVertices.outputVertex;
        yObservationVertex = outputVertices.observedVertex;
        net = new BayesianNetwork(yVertex.getConnectedGraph());
    }

    public OUTPUT predict(DoubleTensor input) {
        xVertex.setAndCascade(input);
        return yVertex.getValue();
    }

    @Override
    public void observeValues(DoubleTensor input, OUTPUT output) {
        xVertex.setValue(input);
        yObservationVertex.observe(output);
    }

    @Value
    public static class OutputVertices<OUTPUT> {
        Vertex<OUTPUT> outputVertex;
        Vertex<OUTPUT> observedVertex;
    }
}
