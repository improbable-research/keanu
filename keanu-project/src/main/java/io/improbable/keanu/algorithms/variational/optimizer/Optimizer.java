package io.improbable.keanu.algorithms.variational.optimizer;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.Vertex;

public interface Optimizer {

    void addFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler);

    void removeFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler);

    double maxAPosteriori();

    double maxLikelihood();

    BayesianNetwork getBayesianNetwork();

    static Optimizer of(BayesianNetwork network) {
        if (network.getDiscreteLatentVertices().isEmpty()) {
            return GradientOptimizer.of(network);
        } else {
            return NonGradientOptimizer.of(network);
        }
    }

    static Optimizer of(Collection<? extends Vertex> vertices) {
        return of(new BayesianNetwork(vertices));
    }

    static Optimizer ofConnectedGraph(Vertex<?> vertexFromNetwork) {
        return of(vertexFromNetwork.getConnectedGraph());
    }

    static double[] currentPoint(List<? extends Vertex<? extends NumberTensor>> continuousLatentVertices) {
        long totalLatentDimensions = totalNumberOfLatentDimensions(continuousLatentVertices);

        if (totalLatentDimensions > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Greater than " + Integer.MAX_VALUE + " latent dimensions not supported");
        }

        int position = 0;
        double[] point = new double[(int) totalLatentDimensions];

        for (Vertex<? extends NumberTensor> vertex : continuousLatentVertices) {
            double[] values = vertex.getValue().asFlatDoubleArray();
            System.arraycopy(values, 0, point, position, values.length);
            position += values.length;
        }

        return point;
    }

    static void setAndCascadePoint(double[] point, List<? extends Vertex<DoubleTensor>> latentVertices) {

        int position = 0;
        for (Vertex<DoubleTensor> vertex : latentVertices) {

            int dimensions = (int) Optimizer.numDimensions(vertex);

            double[] values = new double[dimensions];
            System.arraycopy(point, position, values, 0, dimensions);

            DoubleTensor newTensor = DoubleTensor.create(values, vertex.getValue().getShape());
            vertex.setValue(newTensor);

            position += dimensions;
        }

        VertexValuePropagation.cascadeUpdate(latentVertices);
    }

    static long totalNumberOfLatentDimensions(List<? extends Vertex<? extends NumberTensor>> continuousLatentVertices) {
        return continuousLatentVertices.stream().mapToLong(Optimizer::numDimensions).sum();
    }

    static long numDimensions(Vertex<? extends Tensor> vertex) {
        return vertex.getValue().getLength();
    }

    static ProgressBar createFitnessProgressBar(final Optimizer optimizerThatNeedsProgressBar) {
        AtomicInteger evalCount = new AtomicInteger(0);
        ProgressBar progressBar = new ProgressBar();
        BiConsumer<double[], Double> progressBarFitnessCalculationHandler = (position, logProb) -> {
            progressBar.progress(
                String.format("Fitness Evaluation #%d LogProb: %.2f", evalCount.incrementAndGet(), logProb)
            );
        };

        optimizerThatNeedsProgressBar.addFitnessCalculationHandler(progressBarFitnessCalculationHandler);
        progressBar.addFinishHandler(() -> optimizerThatNeedsProgressBar.removeFitnessCalculationHandler(progressBarFitnessCalculationHandler));

        return progressBar;
    }
}
