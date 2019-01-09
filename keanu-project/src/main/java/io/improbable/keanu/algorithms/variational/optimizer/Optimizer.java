package io.improbable.keanu.algorithms.variational.optimizer;

import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.status.StatusBar;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;

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

    static double[] convertToPoint(List<? extends Variable<? extends NumberTensor>> latentVariables) {

        List<long[]> shapes = latentVariables.stream().map(Variable::getShape).collect(Collectors.toList());

        long totalLatentDimensions = totalNumberOfLatentDimensions(shapes);

        if (totalLatentDimensions > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Greater than " + Integer.MAX_VALUE + " latent dimensions not supported");
        }

        int position = 0;
        double[] point = new double[(int) totalLatentDimensions];

        for (Variable<? extends NumberTensor> variable : latentVariables) {
            double[] values = variable.getValue().asFlatDoubleArray();
            System.arraycopy(values, 0, point, position, values.length);
            position += values.length;
        }

        return point;
    }

    static Map<VariableReference, DoubleTensor> convertFromPoint(double[] point, List<? extends Variable> latentVariables) {

        Map<VariableReference, DoubleTensor> tensors = new HashMap<>();
        int position = 0;
        for (Variable variable : latentVariables) {

            int dimensions = Ints.checkedCast(TensorShape.getLength(variable.getShape()));

            double[] values = new double[dimensions];
            System.arraycopy(point, position, values, 0, dimensions);

            DoubleTensor newTensor = DoubleTensor.create(values, variable.getShape());

            tensors.put(variable.getReference(), newTensor);
            position += dimensions;
        }

        return tensors;
    }

    static long totalNumberOfLatentDimensions(List<long[]> continuousLatentVariableShapes) {
        return continuousLatentVariableShapes.stream().mapToLong(Optimizer::numDimensions).sum();
    }

    static long numDimensions(long[] shape) {
        return TensorShape.getLength(shape);
    }

    static List<Variable<? extends DoubleTensor>> getAsDoubleTensors(List<? extends Variable> variables) {
        return variables.stream()
            .map(
                v -> {
                    if (v.getValue() instanceof DoubleTensor) {
                        return (Variable<DoubleTensor>) v;
                    } else {
                        throw new UnsupportedOperationException(
                            "Optimization unsupported on networks containing discrete latents. " +
                                "Discrete latent : " + v.getReference() + " found.");
                    }
                }
            ).collect(Collectors.toList());
    }

    static StatusBar createFitnessStatusBar(final Optimizer optimizerThatNeedsStatusBar) {
        AtomicInteger evalCount = new AtomicInteger(0);
        StatusBar statusBar = new StatusBar();
        BiConsumer<double[], Double> statusBarFitnessCalculationHandler = (position, logProb) -> {
            statusBar.setMessage(
                String.format("Fitness Evaluation #%d LogProb: %.2f", evalCount.incrementAndGet(), logProb)
            );
        };

        optimizerThatNeedsStatusBar.addFitnessCalculationHandler(statusBarFitnessCalculationHandler);
        statusBar.addFinishHandler(() -> optimizerThatNeedsStatusBar.removeFitnessCalculationHandler(statusBarFitnessCalculationHandler));

        return statusBar;
    }
}
