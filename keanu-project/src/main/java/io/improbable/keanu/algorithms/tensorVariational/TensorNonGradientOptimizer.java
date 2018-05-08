package io.improbable.keanu.algorithms.tensorVariational;

import io.improbable.keanu.network.TensorBayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;

import java.util.List;

import static io.improbable.keanu.algorithms.tensorVariational.TensorGradientOptimizer.currentPoint;
import static org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MAXIMIZE;

public class TensorNonGradientOptimizer {

    private final TensorBayesNet bayesNet;

    public TensorNonGradientOptimizer(TensorBayesNet bayesNet) {
        this.bayesNet = bayesNet;
    }

    public TensorNonGradientOptimizer(List<Vertex<Double>> graph) {
        bayesNet = new TensorBayesNet(graph);
    }

    public double optimize(int maxEvaluations, double boundsRange, List<Vertex> outputVertices) {

        if (bayesNet.isInImpossibleState()) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }

        List<? extends Vertex<DoubleTensor>> latentVertices = bayesNet.getContinuousLatentVertices();
        TensorFitnessFunction fitnessFunction = new TensorFitnessFunction(outputVertices, latentVertices);

        BOBYQAOptimizer optimizer = new BOBYQAOptimizer(2 * latentVertices.size() + 1);

        double[] startPoint = currentPoint(bayesNet.getContinuousLatentVertices());
        double initialFitness = fitnessFunction.fitness().value(startPoint);

        if (TensorFitnessFunction.isValidInitialFitness(initialFitness)) {
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