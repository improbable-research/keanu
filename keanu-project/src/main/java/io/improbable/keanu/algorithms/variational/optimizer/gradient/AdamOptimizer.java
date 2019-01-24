package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.*;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * https://arxiv.org/pdf/1412.6980.pdf
 */
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class AdamOptimizer implements Optimizer {

    public static AdamOptimizerBuilder builder() {
        return new AdamOptimizerBuilder();
    }

    private final ProbabilisticWithGradientGraph probabilisticWithGradientGraph;
    private final double alpha;
    private final double beta1;
    private final double beta2;
    private final double epsilon;

    private final List<BiConsumer<DoubleTensor[], DoubleTensor[]>> onGradientCalculations = new ArrayList<>();

    private OptimizedResult optimize(boolean isMLE) {

        List<? extends Variable> latentVariables = probabilisticWithGradientGraph.getLatentVariables();
        DoubleTensor[] theta = getTheta(latentVariables);
        DoubleTensor[] thetaNext = getZeros(theta);
        DoubleTensor[] m = getZeros(theta);
        DoubleTensor[] v = getZeros(theta);

        int t = 0;
        boolean converged = false;

        int maxIterations = Integer.MAX_VALUE;

        while (!converged && t < maxIterations) {
            t++;

            final DoubleTensor[] gradients = getGradients(theta, latentVariables, isMLE);

            final double beta1T = (1 - Math.pow(beta1, t));
            final double beta2T = (1 - Math.pow(beta2, t));
            final double b = beta1T / Math.sqrt(beta2T);

            for (int i = 0; i < gradients.length; i++) {

                m[i] = m[i].times(beta1).plus(gradients[i].times(1 - beta1));
                v[i] = v[i].times(beta2).plus(gradients[i].pow(2).times(1 - beta2));

                thetaNext[i] = theta[i].plus(m[i].times(alpha).div(v[i].sqrt().times(b).plus(epsilon)));
            }

            converged = hasConverged(gradients);

            final DoubleTensor[] temp = theta;
            theta = thetaNext;
            thetaNext = temp;
        }

        double logProb = probabilisticWithGradientGraph.logProb();

        return new OptimizedResult(toMap(theta, latentVariables), logProb);
    }

    private DoubleTensor[] getGradients(DoubleTensor[] theta, List<? extends Variable> latentVariables, boolean isMLE) {
        Map<VariableReference, DoubleTensor> thetaMap = toMap(theta, latentVariables);

        final DoubleTensor[] gradients;
        if (isMLE) {
            gradients = toArray(
                probabilisticWithGradientGraph.logLikelihoodGradients(thetaMap),
                latentVariables
            );
        } else {
            gradients = toArray(
                probabilisticWithGradientGraph.logProbGradients(thetaMap),
                latentVariables
            );
        }

        handleGradientCalculation(theta, gradients);

        return gradients;
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

    private double magnitude(DoubleTensor[] values) {

        double magPow2 = 0;
        for (int i = 0; i < values.length; i++) {
            magPow2 += values[i].pow(2).sum();
        }

        return Math.sqrt(magPow2);
    }

    private boolean hasConverged(DoubleTensor[] gradient) {
        return magnitude(gradient) < 1e-6;
    }

    @Override
    public OptimizedResult maxAPosteriori() {
        return optimize(false);
    }

    @Override
    public OptimizedResult maxLikelihood() {
        return optimize(true);
    }

    public void addGradientCalculationHandler(BiConsumer<DoubleTensor[], DoubleTensor[]> gradientCalculationHandler) {
        this.onGradientCalculations.add(gradientCalculationHandler);
    }

    public void removeGradientCalculationHandler(BiConsumer<DoubleTensor[], DoubleTensor[]> gradientCalculationHandler) {
        this.onGradientCalculations.remove(gradientCalculationHandler);
    }

    private void handleGradientCalculation(DoubleTensor[] point, DoubleTensor[] gradients) {

        for (BiConsumer<DoubleTensor[], DoubleTensor[]> gradientCalculationHandler : onGradientCalculations) {
            gradientCalculationHandler.accept(point, gradients);
        }
    }

    @Override
    public void addFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler) {
    }

    @Override
    public void removeFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler) {
    }

    public static class AdamOptimizerBuilder {
        private ProbabilisticWithGradientGraph probabilisticWithGradientGraph;

        private double alpha = 0.001;
        private double beta1 = 0.9;
        private double beta2 = 0.999;
        private double epsilon = 1e-8;

        AdamOptimizerBuilder() {
        }

        public AdamOptimizerBuilder bayesianNetwork(ProbabilisticWithGradientGraph probabilisticWithGradientGraph) {
            this.probabilisticWithGradientGraph = probabilisticWithGradientGraph;
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
            return new AdamOptimizer(probabilisticWithGradientGraph, alpha, beta1, beta2, epsilon);
        }

        public String toString() {
            return "AdamOptimizer.AdamOptimizerBuilder(bayesianNetwork=" + this.probabilisticWithGradientGraph + ", alpha=" + this.alpha + ", beta1=" + this.beta1 + ", beta2=" + this.beta2 + ", epsilon=" + this.epsilon + ")";
        }
    }
}
