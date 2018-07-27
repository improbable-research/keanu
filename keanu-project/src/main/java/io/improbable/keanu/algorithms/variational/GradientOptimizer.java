package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import lombok.Builder;
import lombok.Getter;
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

@Builder
public class GradientOptimizer implements Optimizer {

    private static final double FLAT_GRADIENT = 1e-16;

    public static final NonLinearConjugateGradientOptimizer DEFAULT_OPTIMIZER = new NonLinearConjugateGradientOptimizer(
        NonLinearConjugateGradientOptimizer.Formula.POLAK_RIBIERE,
        new SimpleValueChecker(1e-8, 1e-8)
    );

    public static GradientOptimizer of(BayesianNetwork bayesNet) {
        return GradientOptimizer.builder().bayesianNetwork(bayesNet).build();
    }

    @Getter
    private BayesianNetwork bayesianNetwork;

    /**
     * maxEvaluations the maximum number of objective function evaluations before throwing an exception
     * indicating convergence failure.
     */
    @Builder.Default
    private int maxEvaluations = Integer.MAX_VALUE;

    /**
     * apache math optimizer to use for optimization
     */
    @Builder.Default
    private NonLinearConjugateGradientOptimizer optimizer = DEFAULT_OPTIMIZER;

    private double relativeThreshold;
    private double absoluteThreshold;

    private final List<BiConsumer<double[], double[]>> onGradientCalculations = new ArrayList<>();
    private final List<BiConsumer<double[], Double>> onFitnessCalculations = new ArrayList<>();

    public void onGradientCalculation(BiConsumer<double[], double[]> gradientCalculationHandler) {
        this.onGradientCalculations.add(gradientCalculationHandler);
    }

    private void handleGradientCalculation(double[] point, double[] gradients) {
        for (BiConsumer<double[], double[]> gradientCalculationHandler : onGradientCalculations) {
            gradientCalculationHandler.accept(point, gradients);
        }
    }

    @Override
    public void onFitnessCalculation(BiConsumer<double[], Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.add(fitnessCalculationHandler);
    }

    private void handleFitnessCalculation(double[] point, Double fitness) {
        for (BiConsumer<double[], Double> fitnessCalculationHandler : onFitnessCalculations) {
            fitnessCalculationHandler.accept(point, fitness);
        }
    }

    @Override
    public double maxAPosteriori() {
        if (bayesianNetwork.getLatentAndObservedVertices().isEmpty()) {
            throw new IllegalArgumentException("Cannot find MAP of network without any probabilistic vertices");
        }
        return optimize(bayesianNetwork.getLatentAndObservedVertices());
    }

    @Override
    public double maxLikelihood() {
        if (bayesianNetwork.getObservedVertices().isEmpty()) {
            throw new IllegalArgumentException("Cannot find max likelihood of network without any observations");
        }
        return optimize(bayesianNetwork.getObservedVertices());
    }

    private double optimize(List<Vertex> outputVertices) {

        bayesianNetwork.cascadeObservations();

        FitnessFunctionWithGradient fitnessFunction = new FitnessFunctionWithGradient(
            outputVertices,
            bayesianNetwork.getContinuousLatentVertices(),
            this::handleGradientCalculation,
            this::handleFitnessCalculation
        );

        ObjectiveFunction fitness = new ObjectiveFunction(fitnessFunction.fitness());
        ObjectiveFunctionGradient gradient = new ObjectiveFunctionGradient(fitnessFunction.gradient());

        double[] startingPoint = Optimizer.currentPoint(bayesianNetwork.getContinuousLatentVertices());
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

    private static void warnIfGradientIsFlat(double[] gradient) {
        double maxGradient = Arrays.stream(gradient).max().orElseThrow(IllegalArgumentException::new);
        if (Math.abs(maxGradient) <= FLAT_GRADIENT) {
            throw new IllegalStateException("The initial gradient is very flat. The largest gradient is " + maxGradient);
        }
    }
}