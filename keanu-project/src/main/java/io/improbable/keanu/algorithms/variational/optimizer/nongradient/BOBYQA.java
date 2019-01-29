package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.algorithms.variational.optimizer.*;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.ApacheFitnessFunctionAdaptor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.ToString;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;

import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.getAsDoubleTensors;
import static java.util.stream.Collectors.toList;
import static org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MAXIMIZE;

@AllArgsConstructor
public class BOBYQA implements NonGradientOptimizationAlgorithm {

    /**
     * maxEvaluations the maximum number of objective function evaluations before throwing an exception
     * indicating convergence failure.
     */
    private final int maxEvaluations;

    /**
     * bounding box around starting point
     */
    private final double boundsRange;

    /**
     * bounds for each specific continuous latent vertex
     */
    private final OptimizerBounds optimizerBounds;

    /**
     * radius around region to start testing points
     */
    private final double initialTrustRegionRadius;

    /**
     * stopping trust region radius
     */
    private final double stoppingTrustRegionRadius;

    public static BOBYQABuilder builder() {
        return new BOBYQABuilder();
    }

    @Override
    public OptimizedResult optimize(List<? extends Variable> latentVariables, FitnessFunction fitnessFunction) {

        List<long[]> shapes = latentVariables.stream()
            .map(Variable::getShape)
            .collect(toList());

        BOBYQAOptimizer optimizer = new BOBYQAOptimizer(
            getNumInterpolationPoints(shapes),
            initialTrustRegionRadius,
            stoppingTrustRegionRadius
        );

        ObjectiveFunction fitness = new ObjectiveFunction(
            new ApacheFitnessFunctionAdaptor(fitnessFunction, latentVariables)
        );

        double[] startPoint = Optimizer.convertToArrayPoint(getAsDoubleTensors(latentVariables));

        ApacheMathSimpleBoundsCalculator boundsCalculator = new ApacheMathSimpleBoundsCalculator(boundsRange, optimizerBounds);
        SimpleBounds bounds = boundsCalculator.getBounds(latentVariables, startPoint);

        PointValuePair pointValuePair = optimizer.optimize(
            new MaxEval(maxEvaluations),
            fitness,
            bounds,
            MAXIMIZE,
            new InitialGuess(startPoint)
        );

        Map<VariableReference, DoubleTensor> optimizedValues = Optimizer
            .convertFromPoint(pointValuePair.getPoint(), latentVariables);

        return new OptimizedResult(optimizedValues, pointValuePair.getValue());
    }

    private int getNumInterpolationPoints(List<long[]> latentVariableShapes) {
        return (int) (2 * Optimizer.totalNumberOfLatentDimensions(latentVariableShapes) + 1);
    }

    @ToString
    public static class BOBYQABuilder {

        private int maxEvaluations = Integer.MAX_VALUE;
        private double boundsRange = Double.POSITIVE_INFINITY;
        private OptimizerBounds optimizerBounds = new OptimizerBounds();
        private double initialTrustRegionRadius = BOBYQAOptimizer.DEFAULT_INITIAL_RADIUS;
        private double stoppingTrustRegionRadius = BOBYQAOptimizer.DEFAULT_STOPPING_RADIUS;

        BOBYQABuilder() {
        }

        public BOBYQABuilder maxEvaluations(int maxEvaluations) {
            this.maxEvaluations = maxEvaluations;
            return this;
        }

        public BOBYQABuilder boundsRange(double boundsRange) {
            this.boundsRange = boundsRange;
            return this;
        }

        public BOBYQABuilder optimizerBounds(OptimizerBounds optimizerBounds) {
            this.optimizerBounds = optimizerBounds;
            return this;
        }

        public BOBYQABuilder initialTrustRegionRadius(double initialTrustRegionRadius) {
            this.initialTrustRegionRadius = initialTrustRegionRadius;
            return this;
        }

        public BOBYQABuilder stoppingTrustRegionRadius(double stoppingTrustRegionRadius) {
            this.stoppingTrustRegionRadius = stoppingTrustRegionRadius;
            return this;
        }

        public BOBYQA build() {
            return new BOBYQA(maxEvaluations, boundsRange, optimizerBounds, initialTrustRegionRadius, stoppingTrustRegionRadius);
        }
    }
}
