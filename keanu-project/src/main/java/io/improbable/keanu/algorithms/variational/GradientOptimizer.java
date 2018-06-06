package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient;
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiConsumer;

import static org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MAXIMIZE;

public class GradientOptimizer {

    public static final NonLinearConjugateGradientOptimizer DEFAULT_OPTIMIZER = new NonLinearConjugateGradientOptimizer(
        NonLinearConjugateGradientOptimizer.Formula.POLAK_RIBIERE,
        new SimpleValueChecker(1e-8, 1e-8)
    );

    private static final double FLAT_GRADIENT = 1e-16;

    private final BayesianNetwork bayesNet;

    private final List<BiConsumer<double[], double[]>> onGradientCalculations;
    private final List<BiConsumer<double[], Double>> onFitnessCalculations;

    public GradientOptimizer(BayesianNetwork bayesNet) {
        this.bayesNet = bayesNet;
        this.onGradientCalculations = new ArrayList<>();
        this.onFitnessCalculations = new ArrayList<>();
    }

    public void onGradientCalculation(BiConsumer<double[], double[]> gradientCalculationHandler) {
        this.onGradientCalculations.add(gradientCalculationHandler);
    }

    private void handleGradientCalculation(double[] point, double[] gradients) {
        for (BiConsumer<double[], double[]> gradientCalculationHandler : onGradientCalculations) {
            gradientCalculationHandler.accept(point, gradients);
        }
    }

    public void onFitnessCalculation(BiConsumer<double[], Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.add(fitnessCalculationHandler);
    }

    private void handleFitnessCalculation(double[] point, Double fitness) {
        for (BiConsumer<double[], Double> fitnessCalculationHandler : onFitnessCalculations) {
            fitnessCalculationHandler.accept(point, fitness);
        }
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

        bayesNet.cascadeObservations();

        FitnessFunctionWithGradient fitnessFunction = new FitnessFunctionWithGradient(
            outputVertices,
            bayesNet.getContinuousLatentVertices(),
            this::handleGradientCalculation,
            this::handleFitnessCalculation
        );

        ObjectiveFunction fitness = new ObjectiveFunction(fitnessFunction.fitness());
        ObjectiveFunctionGradient gradient = new ObjectiveFunctionGradient(fitnessFunction.gradient());

        double[] startingPoint = currentPoint(bayesNet.getContinuousLatentVertices());
        double initialFitness = fitness.getObjectiveFunction().value(startingPoint);
        double[] initialGradient = gradient.getObjectiveFunctionGradient().value(startingPoint);

        if (FitnessFunction.isValidInitialFitness(initialFitness)) {
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

        long totalLatentDimensions = 0;
        for (Vertex<DoubleTensor> vertex : continuousVertices) {
            totalLatentDimensions += FitnessFunction.numDimensions(vertex);
        }

        if (totalLatentDimensions > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Greater than " + Integer.MAX_VALUE + " latent dimensions not supported");
        }

        int position = 0;
        double[] point = new double[(int) totalLatentDimensions];

        for (Vertex<DoubleTensor> vertex : continuousVertices) {
            double[] values = vertex.getValue().asFlatDoubleArray();
            System.arraycopy(values, 0, point, position, values.length);
            position += values.length;
        }

        return point;
    }

    private void warnIfGradientIsFlat(double[] gradient) {
        double maxGradient = Arrays.stream(gradient).max().getAsDouble();
        if (Math.abs(maxGradient) <= FLAT_GRADIENT) {
            throw new IllegalStateException("The initial gradient is very flat. The largest gradient is " + maxGradient);
        }
    }
}