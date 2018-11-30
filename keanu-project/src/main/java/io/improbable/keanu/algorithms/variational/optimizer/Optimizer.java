package io.improbable.keanu.algorithms.variational.optimizer;

import com.google.common.primitives.Ints;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

import static java.util.stream.Collectors.toMap;

public interface Optimizer {

    /**
     * Adds a callback to be called whenever the optimizer evaluates the fitness of a point. E.g. for logging.
     *
     * @param fitnessCalculationHandler a function to be called whenever the optimizer evaluates the fitness of a point.
     *                                  The double[] argument to the handler represents the point being evaluated.
     *                                  The Double argument to the handler represents the fitness of that point.
     */
    void addFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler);

    /**
     * Removes a callback function that previously would have been called whenever the optimizer
     * evaluated the fitness of a point. If the callback is not registered then this function will do nothing.
     *
     * @param fitnessCalculationHandler the function to be removed from the list of fitness evaluation callbacks
     */
    void removeFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler);

    /**
     * This will use MAP estimation to optimize the values of latent vertices such that the
     * probability of the whole Bayesian network is maximised.
     * This method will modify in place the Bayesian network that was used to create this object.
     *
     * @return the natural logarithm of the Maximum a posteriori (MAP)
     */
    double maxAPosteriori();

    /**
     * This method will use Maximum Likelihood estimation to optimize the values of latent vertices such that
     * the probability of the observed vertices is maximised.
     * This method will modify in place the Bayesian network that was used to create this object.
     *
     * @return the natural logarithm of the maximum likelihood (MLE)
     */
    double maxLikelihood();

    /**
     * Creates an {@link Optimizer} which provides methods for optimizing the values of latent variables
     * of the Bayesian network to maximise probability.
     *
     * @param network The Bayesian network to run optimization on.
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
     * create an {@link Optimizer}. This provides methods for optimizing the values of latent variables
     * of the Bayesian network to maximise probability.
     *
     * @param vertices The vertices to create a Bayesian network from.
     * @return an {@link Optimizer}
     */
    static Optimizer of(Collection<? extends Vertex> vertices) {
        return of(new BayesianNetwork(vertices));
    }

    /**
     * Creates a Bayesian network from the graph connected to the given vertex and uses this to
     * create an {@link Optimizer}. This provides methods for optimizing the values of latent variables
     * of the Bayesian network to maximise probability.
     *
     * @param vertexFromNetwork A vertex in the graph to create the Bayesian network from
     * @return an {@link Optimizer}
     */
    static Optimizer ofConnectedGraph(Vertex<?> vertexFromNetwork) {
        return of(vertexFromNetwork.getConnectedGraph());
    }

    static double[] convertToPoint(List<VariableReference> latentVariables,
                                   Map<VariableReference, ? extends NumberTensor> latentVariableValues,
                                   Map<VariableReference, long[]> latentShapes) {

        long totalLatentDimensions = totalNumberOfLatentDimensions(latentShapes);

        if (totalLatentDimensions > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Greater than " + Integer.MAX_VALUE + " latent dimensions not supported");
        }

        int position = 0;
        double[] point = new double[(int) totalLatentDimensions];

        for (VariableReference variable : latentVariables) {
            double[] values = latentVariableValues.get(variable).asFlatDoubleArray();
            System.arraycopy(values, 0, point, position, values.length);
            position += values.length;
        }

        return point;
    }

    static Map<VariableReference, DoubleTensor> convertFromPoint(double[] point,
                                                                 List<VariableReference> latentVariables,
                                                                 Map<VariableReference, long[]> latentShapes) {

        Map<VariableReference, DoubleTensor> tensors = new HashMap<>();
        int position = 0;
        for (VariableReference variable : latentVariables) {

            long[] latentShape = latentShapes.get(variable);
            int dimensions = Ints.checkedCast(TensorShape.getLength(latentShape));

            double[] values = new double[dimensions];
            System.arraycopy(point, position, values, 0, dimensions);

            DoubleTensor newTensor = DoubleTensor.create(values, latentShape);

            tensors.put(variable, newTensor);
            position += dimensions;
        }

        return tensors;
    }

    static long totalNumberOfLatentDimensions(Map<VariableReference, long[]> continuousLatentVariableShapes) {
        return continuousLatentVariableShapes.values().stream().mapToLong(Optimizer::numDimensions).sum();
    }

    static long numDimensions(long[] shape) {
        return TensorShape.getLength(shape);
    }

    static void initializeNetworkForOptimization(BayesianNetwork bayesianNetwork) {
        List<Vertex> discreteLatentVertices = bayesianNetwork.getDiscreteLatentVertices();
        boolean containsDiscreteLatents = !discreteLatentVertices.isEmpty();

        if (containsDiscreteLatents) {
            throw new UnsupportedOperationException(
                "Gradient optimization unsupported on networks containing discrete latents. " +
                    "Found " + discreteLatentVertices.size() + " discrete latents.");
        }

        bayesianNetwork.cascadeObservations();
    }

    static Map<VariableReference, ? extends NumberTensor> getAsNumberTensors(Map<VariableReference, ?> variableValues) {
        return variableValues.entrySet().stream()
            .collect(toMap(
                Map.Entry::getKey,
                e -> {
                    Object value = e.getValue();
                    if (value instanceof NumberTensor) {
                        return (NumberTensor) e.getValue();
                    } else {
                        throw new UnsupportedOperationException(
                            "Optimization unsupported on networks containing discrete latents. " +
                                "Discrete latent : " + e.getKey() + " found.");
                    }
                }
            ));
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
