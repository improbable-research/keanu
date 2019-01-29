package io.improbable.keanu.model.regression;

import com.google.common.base.Preconditions;
import io.improbable.keanu.model.ModelGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.io.DotSaver;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import java.io.FileOutputStream;
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
        long featureCount = featureShape[1];
        Preconditions.checkArgument(TensorShape.isLengthOne(interceptVertex.getShape()));
        TensorShapeValidation.checkShapesMatch(weightsVertex.getShape(), new long[]{featureCount, 1});

        this.weightsVertex = weightsVertex;
        this.interceptVertex = interceptVertex;
        xVertex = new ConstantDoubleVertex(DoubleTensor.zeros(featureShape));

        OutputVertices<OUTPUT> outputVertices = outputTransform.apply(
            TensorShape.isLengthOne(weightsVertex.getShape()) ?
                weightsVertex.times(xVertex).plus(interceptVertex) :
                xVertex.matrixMultiply(weightsVertex).plus(interceptVertex)
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

    public DoubleVertex getInterceptVertex() {
        return interceptVertex;
    }

    public DoubleVertex getWeightVertex() {
        saveDot(bayesianNetwork, "linearregression.dot");
        return weightsVertex;
    }

    public Vertex<OUTPUT> getOutputVertex() {
        return yObservationVertex;
    }

    private void saveDot(BayesianNetwork net, final String name) {
        try {
            new DotSaver(net).save(new FileOutputStream(name), true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    @Value
    public static class OutputVertices<OUTPUT> {
        Vertex<OUTPUT> outputVertex;
        Vertex<OUTPUT> observedVertex;
    }
}
