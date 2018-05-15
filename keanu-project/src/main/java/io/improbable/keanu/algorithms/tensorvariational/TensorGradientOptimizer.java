package io.improbable.keanu.algorithms.tensorvariational;

import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.TensorBayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient;
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

import static org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MAXIMIZE;

public class TensorGradientOptimizer {

    private final Logger log = LoggerFactory.getLogger(TensorGradientOptimizer.class);

    private static final double FLAT_GRADIENT = 1e-16;

    private final TensorBayesNet bayesNet;

    public TensorGradientOptimizer(TensorBayesNet bayesNet) {
        this.bayesNet = bayesNet;
    }

    /**
     * This method is here to provide more fine grained control of optimization.
     *
     * @param maxEvaluations the maximum number of objective function evaluations before throwing an exception
     *                       indicating convergence failure.
     * @param optimizer      apache math optimizer to use for optimization
     * @return the natural logarithm of the Maximum A Posteriori (MAP)
     */
    public double maxAPosteriori(int maxEvaluations, NonLinearConjugateGradientOptimizer optimizer) {
        if (bayesNet.getLatentAndObservedVertices().isEmpty()) {
            throw new IllegalArgumentException("Cannot find MAP of network without any probabilistic vertices");
        }
        return optimize(maxEvaluations, bayesNet.getLatentAndObservedVertices(), optimizer);
    }

    /**
     * @param maxEvaluations the maximum number of objective function evaluations before throwing an exception
     *                       indicating convergence failure.
     * @return the natural logarithm of the Maximum A Posteriori (MAP)
     */
    public double maxAPosteriori(int maxEvaluations) {
        return maxAPosteriori(maxEvaluations, GradientOptimizer.DEFAULT_OPTIMIZER);
    }

    /**
     * This method is here to provide more fine grained control of optimization.
     *
     * @param maxEvaluations the maximum number of objective function evaluations before throwing an exception
     *                       indicating convergence failure.
     * @param optimizer      apache math optimizer to use for optimization
     * @return the natural logarithm of the maximum likelihood
     */
    public double maxLikelihood(int maxEvaluations, NonLinearConjugateGradientOptimizer optimizer) {
        if (bayesNet.getObservedVertices().isEmpty()) {
            throw new IllegalArgumentException("Cannot find max likelihood of network without any observations");
        }
        return optimize(maxEvaluations, bayesNet.getObservedVertices(), optimizer);
    }

    /**
     * @param maxEvaluations the maximum number of objective function evaluations before throwing an exception
     *                       indicating convergence failure.
     * @return the natural logarithm of the maximum likelihood
     */
    public double maxLikelihood(int maxEvaluations) {
        return maxLikelihood(maxEvaluations, GradientOptimizer.DEFAULT_OPTIMIZER);
    }

    private double optimize(int maxEvaluations,
                            List<Vertex> outputVertices,
                            NonLinearConjugateGradientOptimizer optimizer) {

        TensorFitnessFunctionWithGradient fitnessFunction = new TensorFitnessFunctionWithGradient(outputVertices, bayesNet.getContinuousLatentVertices());
        ObjectiveFunction fitness = new ObjectiveFunction(fitnessFunction.fitness());
        ObjectiveFunctionGradient gradient = new ObjectiveFunctionGradient(fitnessFunction.gradient());

        double[] startingPoint = currentPoint(bayesNet.getContinuousLatentVertices());
        double initialFitness = fitness.getObjectiveFunction().value(startingPoint);
        double[] initialGradient = gradient.getObjectiveFunctionGradient().value(startingPoint);

        if (TensorFitnessFunction.isValidInitialFitness(initialFitness)) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }

        warnIfGradientIsFlat(initialGradient);

        PointValuePair pointValuePair = optimizer.optimize(
                new MaxEval(maxEvaluations),
                fitness,
                gradient,
                MAXIMIZE,
                new InitialGuess(startingPoint)
        );

        return pointValuePair.getValue();
    }

    static double[] currentPoint(List<Vertex<DoubleTensor>> continuousVertices) {

        int totalLatentDimensions = 0;
        for (Vertex<DoubleTensor> vertex : continuousVertices) {
            totalLatentDimensions += TensorFitnessFunction.numDimensions(vertex);
        }

        int position = 0;
        double[] point = new double[totalLatentDimensions];

        for (Vertex<DoubleTensor> vertex : continuousVertices) {
            double[] values = vertex.getValue().getLinearView();
            System.arraycopy(values, 0, point, position, values.length);
            position += values.length;
        }

        return point;
    }

    private void warnIfGradientIsFlat(double[] gradient) {
        double maxGradient = Arrays.stream(gradient).max().getAsDouble();
        if (Math.abs(maxGradient) <= FLAT_GRADIENT) {
            log.warn("The initial gradient is very flat. The largest gradient is {}", maxGradient);
        }
    }
}