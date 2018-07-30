package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import lombok.Builder;
import lombok.Getter;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;

import java.util.*;
import java.util.function.BiConsumer;

import static org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MAXIMIZE;

@Builder
public class NonGradientOptimizer implements Optimizer {

    public static NonGradientOptimizer of(BayesianNetwork bayesNet) {
        return NonGradientOptimizer.builder()
            .bayesianNetwork(bayesNet)
            .build();
    }

    @Getter
    private final BayesianNetwork bayesianNetwork;

    /**
     * maxEvaluations the maximum number of objective function evaluations before throwing an exception
     * indicating convergence failure.
     */
    @Builder.Default
    private int maxEvaluations = Integer.MAX_VALUE;

    /**
     * bounding box around starting point
     */
    @Builder.Default
    private final double scalarBoundsRange = Double.POSITIVE_INFINITY;

    /**
     * bounds for each specific continuous latent vertex
     */
    @Builder.Default
    private final Map<Vertex<DoubleTensor>, SimpleBounds> vertexSpecificBounds = new HashMap<>();

    /**
     * radius around region to start testing points
     */
    @Builder.Default
    double initialTrustRegionRadius = BOBYQAOptimizer.DEFAULT_INITIAL_RADIUS;

    /**
     * stopping trust region radius
     */
    @Builder.Default
    double stoppingTrustRegionRadius = BOBYQAOptimizer.DEFAULT_STOPPING_RADIUS;

    private final List<BiConsumer<double[], Double>> onFitnessCalculations = new ArrayList<>();

    @Override
    public void onFitnessCalculation(BiConsumer<double[], Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.add(fitnessCalculationHandler);
    }

    private void handleFitnessCalculation(double[] point, Double fitness) {
        for (BiConsumer<double[], Double> fitnessCalculationHandler : onFitnessCalculations) {
            fitnessCalculationHandler.accept(point, fitness);
        }
    }

    private double optimize(List<Vertex> outputVertices) {
        bayesianNetwork.cascadeObservations();

        if (bayesianNetwork.isInImpossibleState()) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }

        List<? extends Vertex<DoubleTensor>> latentVertices = vertexSpecificBounds.isEmpty() ?
            bayesianNetwork.getContinuousLatentVertices() :
            new ArrayList<>(vertexSpecificBounds.keySet());

        FitnessFunction fitnessFunction = new FitnessFunction(
            outputVertices,
            latentVertices,
            this::handleFitnessCalculation
        );

        BOBYQAOptimizer optimizer = new BOBYQAOptimizer(
            getNumInterpolationPoints(latentVertices),
            initialTrustRegionRadius,
            stoppingTrustRegionRadius
        );

        double[] startPoint = Optimizer.currentPoint(latentVertices);
        double initialFitness = fitnessFunction.fitness().value(startPoint);

        if (FitnessFunction.isValidInitialFitness(initialFitness)) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }

        SimpleBounds bounds = getBounds(startPoint);

        PointValuePair pointValuePair = optimizer.optimize(
            new MaxEval(maxEvaluations),
            new ObjectiveFunction(fitnessFunction.fitness()),
            bounds,
            MAXIMIZE,
            new InitialGuess(Optimizer.currentPoint(latentVertices))
        );

        return pointValuePair.getValue();
    }

    private SimpleBounds getBounds(double[] startPoint) {
        if (vertexSpecificBounds.isEmpty()) {
            return getScalarBounds(startPoint);
        }
        return getVertexSpecificBounds();
    }

    private SimpleBounds getScalarBounds(double[] startPoint) {
        double[] minBounds = new double[startPoint.length];
        double[] maxBounds = new double[startPoint.length];

        for (int i = 0; i < startPoint.length; i++) {
            minBounds[i] = startPoint[i] - scalarBoundsRange;
            maxBounds[i] = startPoint[i] + scalarBoundsRange;
        }
        return new SimpleBounds(minBounds, maxBounds);
    }

    private SimpleBounds getVertexSpecificBounds() {
        List<Double> minBounds = new ArrayList<>();
        List<Double> maxBounds = new ArrayList<>();

        vertexSpecificBounds.forEach((vertex, bounds) -> {
            minBounds.addAll(DoubleTensor.create(bounds.getLower()).asFlatList());
            maxBounds.addAll(DoubleTensor.create(bounds.getUpper()).asFlatList());
        });

        return new SimpleBounds(
            minBounds.stream().mapToDouble(d -> d).toArray(),
            maxBounds.stream().mapToDouble(d -> d).toArray()
        );
    }

    private int getNumInterpolationPoints(List<? extends Vertex<DoubleTensor>> latentVertices) {
        return (int) (2 * Optimizer.totalNumberOfLatentDimensions(latentVertices) + 1);
    }

    /**
     * @return the natural logarithm of the Maximum a posteriori (MAP)
     */
    @Override
    public double maxAPosteriori() {
        return optimize(bayesianNetwork.getLatentAndObservedVertices());
    }

    /**
     * @return the natural logarithm of the maximum likelihood (MLE)
     */
    @Override
    public double maxLikelihood() {
        return optimize(bayesianNetwork.getObservedVertices());
    }

}