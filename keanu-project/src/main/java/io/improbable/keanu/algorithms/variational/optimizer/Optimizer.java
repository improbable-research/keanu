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
    /**
     * Adds a callback to be called whenever the optimizer evaluates the fitness of a point.
     * @param fitnessCalculationHandler a function to be called whenever the optimizer evaluates the fitness of a point.
     *                                  The double[] argument to the handler represents the point being evaluated.
     *                                  The Double argument to the handler represents the fitness of that point.
     */
    void addFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler);

    /**
     * This attempts to remove a callback function that previously would have been called whenever the optimizer
     * evaluated the fitness of a point. If the callback is not registered then this function will do nothing.
     * @param fitnessCalculationHandler the function to be removed from the list of fitness evaluation callbacks
     */
    void removeFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler);

    /**
     * This methods detects the vertices in the Bayesian Network that have been observed, and will use MAP estimation to
     * optimize the observation probability of these vertices.
     * This method will modify in place the Bayesian network that was used to create this object.
     *
     * @return the natural logarithm of the Maximum a posteriori (MAP)
     */
    double maxAPosteriori();

    /**
     * This methods detects the vertices in the Bayesian Network that have been observed, and will use Maximum Likelihood
     * estimation to optimize the observation probability of these vertices.
     * This method will modify in place the Bayesian network that was used to create this object.
     *
     * @return the natural logarithm of the maximum likelihood (MLE)
     */
    double maxLikelihood();

    BayesianNetwork getBayesianNetwork();

    /**
     * Creates an {@link Optimizer} which optimizes against the unobserved (latent) variables of the Bayesian network.
     *
     * @param network The Bayesian network containing the unobserved variables to be optimized against,
     *                 and the observed variables to optimize the probability of.
     * @return an {@link Optimizer}
     */
    static Optimizer of(BayesianNetwork network) {
        if (network.getDiscreteLatentVertices().isEmpty()) {
            return GradientOptimizer.of(network);
        } else {
            return NonGradientOptimizer.of(network);
        }
    }

    /**
     * Creates a Bayesian network from the given vertices and uses this to
     * create an {@link Optimizer} which optimizes against the unobserved (latent) variables.
     *
     * @param vertices The vertices to create a Bayesian network from.
     * @return an {@link Optimizer}
     */
    static Optimizer of(Collection<? extends Vertex> vertices) {
        return of(new BayesianNetwork(vertices));
    }

    /**
     * Retrieves the connected graph from the given vertex and uses this to
     * create an {@link Optimizer} which optimizes against the unobserved (latent) variables.
     *
     * @param vertexFromNetwork The vertices to create a Bayesian network from.
     * @return an {@link Optimizer}
     */
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
