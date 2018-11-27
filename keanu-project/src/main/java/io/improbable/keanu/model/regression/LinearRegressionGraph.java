package io.improbable.keanu.model.regression;

import com.google.common.base.Preconditions;
import io.improbable.keanu.model.ModelGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import lombok.Getter;
import lombok.Value;

import java.util.function.Function;

public class LinearRegressionGraph<OUTPUT> implements ModelGraph<DoubleTensor, OUTPUT> {
    private final DoubleVertex xVertex;
    private final Vertex<OUTPUT> yVertex;
    private final Vertex<OUTPUT> yObservationVertex;
    private final DoubleVertex weightsVertex;
    private final DoubleVertex interceptVertex;
    @Getter
    private final BayesianNetwork bayesianNetwork;

    public LinearRegressionGraph(long[] featureShape, Function<DoubleVertex, OutputVertices<OUTPUT>> outputTransform, DoubleVertex interceptVertex, DoubleVertex weightsVertex) {
        long featureCount = featureShape[0];
        Preconditions.checkArgument(TensorShape.isLengthOne(interceptVertex.getShape()));
        TensorShapeValidation.checkShapesMatch(weightsVertex.getShape(), new long[]{1, featureCount});

        this.weightsVertex = weightsVertex;
        this.interceptVertex = interceptVertex;
        xVertex = new ConstantDoubleVertex(DoubleTensor.zeros(featureShape));

        OutputVertices<OUTPUT> outputVertices = outputTransform.apply(
            TensorShape.isLengthOne(weightsVertex.getShape()) ?
                weightsVertex.times(xVertex).plus(interceptVertex) :
                weightsVertex.matrixMultiply(xVertex).plus(interceptVertex)
        );

        yVertex = outputVertices.outputVertex;
        yObservationVertex = outputVertices.observedVertex;
        bayesianNetwork = new BayesianNetwork(yVertex.getConnectedGraph());
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

    public DoubleTensor getWeights() {
        return weightsVertex.getValue();
    }

    public DoubleVertex getInterceptVertex() {
        return interceptVertex;
    }

    public DoubleVertex getWeightVertex() {
        return weightsVertex;
    }

    public double getIntercept() {
        return interceptVertex.getValue().scalar();
    }

    public VertexId getInterceptVertexId() {
        return interceptVertex.getId();
    }

    public VertexId getWeightsVertexId() {
        return weightsVertex.getId();
    }

    @Value
    public static class OutputVertices<OUTPUT> {
        Vertex<OUTPUT> outputVertex;
        Vertex<OUTPUT> observedVertex;
    }
}
