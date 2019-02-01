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
    private final int maxIterations;
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

        final Map<VariableReference, DoubleTensor> thetaMap = new HashMap<>();
        final DoubleTensor[] gradients = new DoubleTensor[theta.length];

        double beta1T = 1;
        double beta2T = 1;

        for (int t = 1; !converged && t <= maxIterations; t++) {

            updateGradients(theta, thetaMap, gradients, latentVariables, fitnessFunctionGradient);

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

        double logProb = fitnessFunction.getFitnessAt(updateMap(theta, latentVariables, thetaMap));

        return new OptimizedResult(updateMap(theta, latentVariables, thetaMap), logProb);
    }

    private void updateGradients(DoubleTensor[] theta,
                                 Map<VariableReference, DoubleTensor> thetaMap,
                                 DoubleTensor[] gradients,
                                 List<? extends Variable> latentVariables,
                                 FitnessFunctionGradient fitnessFunctionGradient) {
        updateMap(theta, latentVariables, thetaMap);

        updateArray(
            fitnessFunctionGradient.getGradientsAt(thetaMap),
            latentVariables,
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

    private DoubleTensor[] updateArray(Map<? extends VariableReference, DoubleTensor> lookup,
                                       List<? extends Variable> ordered,
                                       DoubleTensor[] array) {

        for (int i = 0; i < ordered.size(); i++) {
            array[i] = lookup.get(ordered.get(i).getReference());
        }

        return array;
    }

    private Map<VariableReference, DoubleTensor> updateMap(DoubleTensor[] values,
                                                           List<? extends Variable> ordered,
                                                           Map<VariableReference, DoubleTensor> asMap) {

        for (int i = 0; i < values.length; i++) {
            asMap.put(ordered.get(i).getReference(), values[i]);
        }

        return asMap;
    }


    @ToString
    public static class AdamBuilder {
        private ConvergenceChecker convergenceChecker = new RelativeConvergenceChecker(ConvergenceChecker.Norm.MAX_ABS, 1e-6);

        private int maxIterations = Integer.MAX_VALUE;
        private double alpha = 0.001;
        private double beta1 = 0.9;
        private double beta2 = 0.999;
        private double epsilon = 1e-8;

        public AdamBuilder maxIterations(int maxIterations) {
            this.maxIterations = maxIterations;
            return this;
        }

        public AdamBuilder convergenceChecker(ConvergenceChecker convergenceChecker) {
            this.convergenceChecker = convergenceChecker;
            return this;
        }

        public AdamBuilder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public AdamBuilder beta1(double beta1) {
            this.beta1 = beta1;
            return this;
        }

        public AdamBuilder beta2(double beta2) {
            this.beta2 = beta2;
            return this;
        }

        public AdamBuilder epsilon(double epsilon) {
            this.epsilon = epsilon;
            return this;
        }

        public Adam build() {
            return new Adam(convergenceChecker, maxIterations, alpha, beta1, beta2, epsilon);
        }
    }
}
