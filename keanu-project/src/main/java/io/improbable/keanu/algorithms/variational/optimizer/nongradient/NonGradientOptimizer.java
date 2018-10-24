package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import static org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MAXIMIZE;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.BiConsumer;

import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;

import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.Vertex;
import lombok.Builder;
import lombok.Getter;

/**
 * This class can be used to construct a BOBYQA non-gradient optimizer.
 * This will use a quadratic approximation of the gradient to perform optimization without derivatives.
 * @see <a href="http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf">BOBYQA Optimizer</a>
 */
@Builder
public class NonGradientOptimizer implements Optimizer {
    /**
     * Creates a BOBYQA {@link NonGradientOptimizer} which optimizes the values of the latent variables of
     * the Bayesian network to maximise overall probability.
     *
     * @param bayesNet The Bayesian network to run optimization on.
     * @return a {@link NonGradientOptimizer}
     */
    public static NonGradientOptimizer of(BayesianNetwork bayesNet) {
        return NonGradientOptimizer.builder()
            .bayesianNetwork(bayesNet)
            .build();
    }

    /**
     * Creates a Bayesian network from the given vertices and uses this to
     * create a BOBYQA {@link NonGradientOptimizer} which optimizes the values of latent variables
     * to maximise overall probability.
     *
     * @param vertices The vertices to create a Bayesian network from.
     * @return a {@link NonGradientOptimizer}
     */
    public static NonGradientOptimizer of(Collection<? extends Vertex> vertices) {
        return of(new BayesianNetwork(vertices));
    }

    /**
     * Creates a Bayesian network from the graph connected to the given vertex and uses this to
     * create a BOBYQA {@link NonGradientOptimizer} which optimizes the values of latent variables to maximise overall probability.
     *
     * @param vertexFromNetwork A vertex in the graph to create the Bayesian network from
     * @return a {@link NonGradientOptimizer}
     */
    public static NonGradientOptimizer ofConnectedGraph(Vertex<?> vertexFromNetwork) {
        return of(vertexFromNetwork.getConnectedGraph());
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
    private final double boundsRange = Double.POSITIVE_INFINITY;

    /**
     * bounds for each specific continuous latent vertex
     */
    @Builder.Default
    private final OptimizerBounds optimizerBounds = new OptimizerBounds();

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
    public void addFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.add(fitnessCalculationHandler);
    }

    @Override
    public void removeFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.remove(fitnessCalculationHandler);
    }

    private void handleFitnessCalculation(double[] point, Double fitness) {
        for (BiConsumer<double[], Double> fitnessCalculationHandler : onFitnessCalculations) {
            fitnessCalculationHandler.accept(point, fitness);
        }
    }


    private double optimize(List<Vertex> outputVertices) {

        ProgressBar progressBar = Optimizer.createFitnessProgressBar(this);
        bayesianNetwork.cascadeObservations();

        if (bayesianNetwork.isInImpossibleState()) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }

        List<? extends Vertex<DoubleTensor>> latentVertices = bayesianNetwork.getContinuousLatentVertices();

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

        ApacheMathSimpleBoundsCalculator boundsCalculator = new ApacheMathSimpleBoundsCalculator(boundsRange, optimizerBounds);
        SimpleBounds bounds = boundsCalculator.getBounds(latentVertices, startPoint);

        PointValuePair pointValuePair = optimizer.optimize(
            new MaxEval(maxEvaluations),
            new ObjectiveFunction(fitnessFunction.fitness()),
            bounds,
            MAXIMIZE,
            new InitialGuess(Optimizer.currentPoint(latentVertices))
        );

        progressBar.finish();
        return pointValuePair.getValue();
    }

    private int getNumInterpolationPoints(List<? extends Vertex<DoubleTensor>> latentVertices) {
        return (int) (2 * Optimizer.totalNumberOfLatentDimensions(latentVertices) + 1);
    }

    @Override
    public double maxAPosteriori() {
        return optimize(bayesianNetwork.getLatentOrObservedVertices());
    }

    @Override
    public double maxLikelihood() {
        return optimize(bayesianNetwork.getObservedVertices());
    }

}