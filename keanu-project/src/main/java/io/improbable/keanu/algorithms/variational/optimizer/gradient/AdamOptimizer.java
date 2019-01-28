package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.*;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implemented as described in https://arxiv.org/pdf/1412.6980.pdf
 */
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class AdamOptimizer implements GradientOptimizationAlgorithm {

    public interface ConvergenceChecker {
        boolean hasConverged(DoubleTensor[] gradient, DoubleTensor[] theta, DoubleTensor[] thetaNext);
    }

    public static AdamOptimizerBuilder builder() {
        return new AdamOptimizerBuilder();
    }

    private final ConvergenceChecker convergenceChecker;
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

        int t = 0;
        boolean converged = false;

        int maxIterations = Integer.MAX_VALUE;

        while (!converged && t < maxIterations) {
            t++;

            final DoubleTensor[] gradients = getGradients(theta, latentVariables, fitnessFunctionGradient);

            final double beta1T = (1 - Math.pow(beta1, t));
            final double beta2T = (1 - Math.pow(beta2, t));
            final double b = beta1T / Math.sqrt(beta2T);

            for (int i = 0; i < gradients.length; i++) {

                m[i] = m[i].times(beta1).plus(gradients[i].times(1 - beta1));
                v[i] = v[i].times(beta2).plus(gradients[i].pow(2).times(1 - beta2));

                thetaNext[i] = theta[i].plus(m[i].times(alpha).div(v[i].sqrt().times(b).plus(epsilon)));
            }

            converged = convergenceChecker.hasConverged(gradients, theta, thetaNext);

            final DoubleTensor[] temp = theta;
            theta = thetaNext;
            thetaNext = temp;
        }

        double logProb = fitnessFunction.value(toMap(theta, latentVariables));

        return new OptimizedResult(toMap(theta, latentVariables), logProb);
    }

    private DoubleTensor[] getGradients(DoubleTensor[] theta,
                                        List<? extends Variable> latentVariables,
                                        FitnessFunctionGradient fitnessFunctionGradient) {
        Map<VariableReference, DoubleTensor> thetaMap = toMap(theta, latentVariables);

        return toArray(
            fitnessFunctionGradient.value(thetaMap),
            latentVariables
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

    private DoubleTensor[] toArray(Map<? extends VariableReference, DoubleTensor> lookup, List<? extends Variable> orderded) {

        DoubleTensor[] array = new DoubleTensor[orderded.size()];
        for (int i = 0; i < orderded.size(); i++) {
            array[i] = lookup.get(orderded.get(i).getReference());
        }

        return array;
    }

    private Map<VariableReference, DoubleTensor> toMap(DoubleTensor[] values, List<? extends Variable> orderded) {

        Map<VariableReference, DoubleTensor> asMap = new HashMap<>();
        for (int i = 0; i < values.length; i++) {
            asMap.put(orderded.get(i).getReference(), values[i]);
        }

        return asMap;
    }

    private static double magnitudeDelta(DoubleTensor[] a, DoubleTensor[] b) {
        double magPow2 = 0;
        for (int i = 0; i < a.length; i++) {
            magPow2 += a[i].minus(b[i]).pow(2).sum();
        }

        return Math.sqrt(magPow2);
    }

    private static ConvergenceChecker thetaDeltaMagnitude(final double minThetaDelta) {
        return (gradient, theta, thetaNext) -> magnitudeDelta(theta, thetaNext) < minThetaDelta;
    }

    public static class AdamOptimizerBuilder {
        private ConvergenceChecker convergenceChecker = AdamOptimizer.thetaDeltaMagnitude(1e-6);

        private double alpha = 0.001;
        private double beta1 = 0.9;
        private double beta2 = 0.999;
        private double epsilon = 1e-8;

        AdamOptimizerBuilder() {
        }

        public AdamOptimizerBuilder convergenceChecker(ConvergenceChecker convergenceChecker) {
            this.convergenceChecker = convergenceChecker;
            return this;
        }

        public AdamOptimizerBuilder useMaxThetaDeltaConvergenceChecker(double maxThetaDelta) {
            this.convergenceChecker = AdamOptimizer.thetaDeltaMagnitude(maxThetaDelta);
            return this;
        }

        public AdamOptimizerBuilder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public AdamOptimizerBuilder beta1(double beta1) {
            this.beta1 = beta1;
            return this;
        }

        public AdamOptimizerBuilder beta2(double beta2) {
            this.beta2 = beta2;
            return this;
        }

        public AdamOptimizerBuilder epsilon(double epsilon) {
            this.epsilon = epsilon;
            return this;
        }

        public AdamOptimizer build() {
            return new AdamOptimizer(convergenceChecker, alpha, beta1, beta2, epsilon);
        }
    }
}
