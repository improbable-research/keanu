package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.ConvergenceChecker;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.RelativeConvergenceChecker;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.ToString;
import org.apache.commons.math3.exception.NotStrictlyPositiveException;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implemented as described in https://arxiv.org/pdf/1412.6980.pdf
 */
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class Adam implements GradientOptimizationAlgorithm {

    public static AdamBuilder builder() {
        return new AdamBuilder();
    }

    private final ConvergenceChecker convergenceChecker;
    private final int maxEvaluations;
    private final double alpha;
    private final double beta1;
    private final double beta2;
    private final double epsilon;

    @Override
    public OptimizedResult optimize(List<? extends Variable> latentVariables,
                                    FitnessFunction fitnessFunction,
                                    FitnessFunctionGradient fitnessFunctionGradient) {

        DoubleTensor[] theta = getTheta(latentVariables);
        DoubleTensor[] thetaNext = getZeros(theta);
        DoubleTensor[] m = getZeros(theta);
        DoubleTensor[] v = getZeros(theta);

        boolean converged = false;

        final Map<VariableReference, DoubleTensor> thetaAsPoint = new HashMap<>();
        final DoubleTensor[] gradients = new DoubleTensor[theta.length];

        double beta1T = 1;
        double beta2T = 1;

        for (int t = 1; !converged && t <= maxEvaluations; t++) {

            updateGradients(latentVariables, theta, thetaAsPoint, gradients, fitnessFunctionGradient);

            beta1T = beta1T * beta1;
            beta2T = beta2T * beta2;

            final double b = (1 - beta1T) / Math.sqrt(1 - Math.pow(beta2, t));

            for (int i = 0; i < theta.length; i++) {

                m[i] = m[i].times(beta1).plusInPlace(gradients[i].times(1 - beta1));
                v[i] = v[i].times(beta2).plusInPlace(gradients[i].pow(2).timesInPlace(1 - beta2));

                thetaNext[i] = theta[i].plus(m[i].times(alpha).divInPlace(v[i].sqrt().timesInPlace(b).plusInPlace(epsilon)));
            }

            converged = convergenceChecker.hasConverged(theta, thetaNext);

            final DoubleTensor[] temp = theta;
            theta = thetaNext;
            thetaNext = temp;
        }

        updatePoint(latentVariables, theta, thetaAsPoint);

        double logProb = fitnessFunction.getFitnessAt(thetaAsPoint);

        return new OptimizedResult(thetaAsPoint, logProb);
    }

    private void updateGradients(List<? extends Variable> ordered,
                                 DoubleTensor[] theta,
                                 Map<VariableReference, DoubleTensor> thetaAsPoint,
                                 DoubleTensor[] gradients,
                                 FitnessFunctionGradient fitnessFunctionGradient) {
        updatePoint(
            ordered,
            theta,
            thetaAsPoint
        );

        updateGradients(
            ordered,
            fitnessFunctionGradient.getGradientsAt(thetaAsPoint),
            gradients
        );
    }

    private DoubleTensor[] getTheta(List<? extends Variable> latentVertices) {

        DoubleTensor[] theta = new DoubleTensor[latentVertices.size()];
        for (int i = 0; i < theta.length; i++) {
            theta[i] = (DoubleTensor) latentVertices.get(i).getValue();
        }

        return theta;
    }

    private DoubleTensor[] getZeros(DoubleTensor[] values) {

        DoubleTensor[] zeros = new DoubleTensor[values.length];
        for (int i = 0; i < zeros.length; i++) {
            zeros[i] = DoubleTensor.zeros(values[i].getShape());
        }

        return zeros;
    }

    private void updateGradients(List<? extends Variable> ordered,
                                 Map<? extends VariableReference, DoubleTensor> newGradients,
                                 DoubleTensor[] gradients) {

        for (int i = 0; i < ordered.size(); i++) {
            gradients[i] = newGradients.get(ordered.get(i).getReference());
        }

    }

    private void updatePoint(List<? extends Variable> ordered,
                             DoubleTensor[] values,
                             Map<VariableReference, DoubleTensor> valuesAsPoint) {

        for (int i = 0; i < values.length; i++) {
            valuesAsPoint.put(ordered.get(i).getReference(), values[i]);
        }

    }


    @ToString
    public static class AdamBuilder {
        private ConvergenceChecker convergenceChecker = new RelativeConvergenceChecker(ConvergenceChecker.Norm.L2, 1e-6);

        private int maxEvaluations = Integer.MAX_VALUE;
        private double alpha = 0.001;
        private double beta1 = 0.9;
        private double beta2 = 0.999;
        private double epsilon = 1e-8;

        public AdamBuilder maxEvaluations(int maxEvaluations) {
            if (maxEvaluations <= 0) {
                throw new NotStrictlyPositiveException(maxEvaluations);
            }
            this.maxEvaluations = maxEvaluations;
            return this;
        }

        public AdamBuilder convergenceChecker(ConvergenceChecker convergenceChecker) {
            this.convergenceChecker = convergenceChecker;
            return this;
        }

        public AdamBuilder alpha(double alpha) {
            if (alpha <= 0) {
                throw new NotStrictlyPositiveException(alpha);
            }
            this.alpha = alpha;
            return this;
        }

        public AdamBuilder beta1(double beta1) {
            if (beta1 < 0 || beta1 >= 1) {
                throw new IllegalArgumentException("beta1 must be between 0 (inclusive) and 1 (exclusive)");
            }
            this.beta1 = beta1;
            return this;
        }

        public AdamBuilder beta2(double beta2) {
            if (beta2 < 0 || beta2 >= 1) {
                throw new IllegalArgumentException("beta2 must be between 0 (inclusive) and 1 (exclusive)");
            }
            this.beta2 = beta2;
            return this;
        }

        public AdamBuilder epsilon(double epsilon) {
            if (epsilon <= 0) {
                throw new NotStrictlyPositiveException(epsilon);
            }
            this.epsilon = epsilon;
            return this;
        }

        public Adam build() {
            return new Adam(convergenceChecker, maxEvaluations, alpha, beta1, beta2, epsilon);
        }
    }
}
