package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.Vertex;
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

public class GradientOptimizer {

    private final Logger log = LoggerFactory.getLogger(GradientOptimizer.class);

    private static final NonLinearConjugateGradientOptimizer DEFAULT_OPTIMIZER = new NonLinearConjugateGradientOptimizer(
            NonLinearConjugateGradientOptimizer.Formula.POLAK_RIBIERE,
            new SimpleValueChecker(1e-8, 1e-8)
    );

    private static final double FLAT_GRADIENT = 1e-16;

    private final BayesNet bayesNet;

    public GradientOptimizer(BayesNet bayesNet) {
        this.bayesNet = bayesNet;
    }

    public GradientOptimizer(List<Vertex<Double>> graph) {
        bayesNet = new BayesNet(graph);
    }

    /**
     * @param maxEvaluations throws an exception if the optimizer doesn't converge within this many evaluations
     * @return the natural logarithm of the Maximum a posteriori (MAP)
     */
    public double maxAPosteriori(int maxEvaluations, NonLinearConjugateGradientOptimizer optimizer) {
        if (bayesNet.getVerticesThatContributeToMasterP().isEmpty()) {
            throw new IllegalArgumentException("Cannot find MAP of network without any probabilistic vertices");
        }
        return optimize(maxEvaluations, bayesNet.getVerticesThatContributeToMasterP(), optimizer);
    }

    public double maxAPosteriori(int maxEvaluations) {
        return maxAPosteriori(maxEvaluations, DEFAULT_OPTIMIZER);
    }

    /**
     * @param maxEvaluations throws an exception if the optimizer doesn't converge within this many evaluations
     * @return the natural logarithm of the maximum likelihood
     */
    public double maxLikelihood(int maxEvaluations, NonLinearConjugateGradientOptimizer optimizer) {
        if (bayesNet.getObservedVertices().isEmpty()) {
            throw new IllegalArgumentException("Cannot find max likelihood of network without any observations");
        }
        return optimize(maxEvaluations, bayesNet.getObservedVertices(), optimizer);
    }

    public double maxLikelihood(int maxEvaluations) {
        return maxLikelihood(maxEvaluations, DEFAULT_OPTIMIZER);
    }

    private double optimize(int maxEvaluations,
                            List<Vertex<?>> outputVertices,
                            NonLinearConjugateGradientOptimizer optimizer) {

        FitnessFunctionWithGradient fitnessFunction = new FitnessFunctionWithGradient(outputVertices, bayesNet.getContinuousLatentVertices());
        ObjectiveFunction fitness = new ObjectiveFunction(fitnessFunction.fitness());
        ObjectiveFunctionGradient gradient = new ObjectiveFunctionGradient(fitnessFunction.gradient());

        double[] startingPoint = currentPoint();
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

    private double[] currentPoint() {
        double[] point = new double[bayesNet.getContinuousLatentVertices().size()];
        for (int i = 0; i < point.length; i++) {
            point[i] = bayesNet.getContinuousLatentVertices().get(i).getValue();
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