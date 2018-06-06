package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;

import static io.improbable.keanu.algorithms.variational.GradientOptimizer.currentPoint;
import static org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MAXIMIZE;

public class NonGradientOptimizer {

    private final BayesianNetwork bayesNet;
    private final List<BiConsumer<double[], Double>> onFitnessCalculations;

    public NonGradientOptimizer(BayesianNetwork bayesNet) {
        this.bayesNet = bayesNet;
        this.onFitnessCalculations = new ArrayList<>();
    }

    public NonGradientOptimizer(List<Vertex<Double>> graph) {
        this(new BayesianNetwork(graph));
    }

    public void onFitnessCalculation(BiConsumer<double[], Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.add(fitnessCalculationHandler);
    }

    private void handleFitnessCalculation(double[] point, Double fitness) {
        for (BiConsumer<double[], Double> fitnessCalculationHandler : onFitnessCalculations) {
            fitnessCalculationHandler.accept(point, fitness);
        }
    }

    public double optimize(int maxEvaluations, double boundsRange, List<Vertex> outputVertices) {

        bayesNet.cascadeObservations();

        if (bayesNet.isInImpossibleState()) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }

        List<? extends Vertex<DoubleTensor>> latentVertices = bayesNet.getContinuousLatentVertices();
        FitnessFunction fitnessFunction = new FitnessFunction(
            outputVertices,
            latentVertices,
            this::handleFitnessCalculation
        );

        BOBYQAOptimizer optimizer = new BOBYQAOptimizer(2 * latentVertices.size() + 1);

        double[] startPoint = currentPoint(bayesNet.getContinuousLatentVertices());
        double initialFitness = fitnessFunction.fitness().value(startPoint);

        if (FitnessFunction.isValidInitialFitness(initialFitness)) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }

        double[] minBounds = new double[startPoint.length];
        double[] maxBounds = new double[startPoint.length];

        for (int i = 0; i < startPoint.length; i++) {
            minBounds[i] = startPoint[i] - boundsRange;
            maxBounds[i] = startPoint[i] + boundsRange;
        }

        PointValuePair pointValuePair = optimizer.optimize(
            new MaxEval(maxEvaluations),
            new ObjectiveFunction(fitnessFunction.fitness()),
            new SimpleBounds(minBounds, maxBounds),
            MAXIMIZE,
            new InitialGuess(currentPoint(bayesNet.getContinuousLatentVertices()))
        );

        return pointValuePair.getValue();
    }

    /**
     * @param maxEvaluations throws an exception if the optimizer doesn't converge within this many evaluations
     * @param boundsRange    bounding box around starting point
     * @return the natural logarithm of the Maximum a posteriori (MAP)
     */
    public double maxAPosteriori(int maxEvaluations, double boundsRange) {
        return optimize(maxEvaluations, boundsRange, bayesNet.getLatentAndObservedVertices());
    }

    /**
     * @param maxEvaluations throws an exception if the optimizer doesn't converge within this many evaluations
     * @param boundsRange    bounding box around starting point
     * @return the natural logarithm of the maximum likelihood
     */
    public double maxLikelihood(int maxEvaluations, double boundsRange) {
        return optimize(maxEvaluations, boundsRange, bayesNet.getObservedVertices());
    }

}