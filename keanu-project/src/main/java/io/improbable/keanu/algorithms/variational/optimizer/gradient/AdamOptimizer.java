package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;

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

    private final BayesianNetwork bayesianNetwork;

    private final double alpha;
    private final double beta1;
    private final double beta2;
    private final double epsilon;

    private OptimizedResult optimize(LogProbGradientCalculator gradientCalculator) {

        List<Vertex<DoubleTensor>> latentVariables = bayesianNetwork.getContinuousLatentVertices();
        DoubleTensor[] theta = getTheta(latentVariables);
        DoubleTensor[] thetaNext = getZeros(theta);
        DoubleTensor[] m = getZeros(theta);
        DoubleTensor[] v = getZeros(theta);

        int t = 0;
        boolean converged = false;

        int maxIterations = Integer.MAX_VALUE;

        while (!converged && t < maxIterations) {
            t++;

            setTheta(theta, latentVariables);
            DoubleTensor[] gradientT = toArray(gradientCalculator.getJointLogProbGradientWrtLatents(), latentVariables);

            final double beta1T = (1 - Math.pow(beta1, t));
            final double beta2T = (1 - Math.pow(beta2, t));
            final double b = beta1T / Math.sqrt(beta2T);

            for (int i = 0; i < gradientT.length; i++) {

                m[i] = m[i].times(beta1).plus(gradientT[i].times(1 - beta1));
                v[i] = v[i].times(beta2).plus(gradientT[i].pow(2).times(1 - beta2));

                thetaNext[i] = theta[i].plus(m[i].times(alpha).div(v[i].sqrt().times(b).plus(epsilon)));
            }

            converged = hasConverged(gradientT);

            DoubleTensor[] temp = theta;
            theta = thetaNext;
            thetaNext = temp;
        }

        double logProb = bayesianNetwork.getLogOfMasterP();

        return new OptimizedResult(toMap(theta, latentVariables), logProb);
    }

    private DoubleTensor[] getTheta(List<Vertex<DoubleTensor>> latentVertices) {

        DoubleTensor[] theta = new DoubleTensor[latentVertices.size()];
        for (int i = 0; i < theta.length; i++) {
            theta[i] = latentVertices.get(i).getValue();
        }

        return theta;
    }

    private void setTheta(DoubleTensor[] theta, List<Vertex<DoubleTensor>> latentVertices) {

        for (int i = 0; i < theta.length; i++) {
            latentVertices.get(i).setValue(theta[i]);
        }
        VertexValuePropagation.cascadeUpdate(latentVertices);
    }

    private DoubleTensor[] getZeros(DoubleTensor[] values) {

        DoubleTensor[] zeros = new DoubleTensor[values.length];
        for (int i = 0; i < zeros.length; i++) {
            zeros[i] = DoubleTensor.zeros(values[i].getShape());
        }

        return zeros;
    }

    private DoubleTensor[] toArray(Map<VertexId, DoubleTensor> lookup, List<Vertex<DoubleTensor>> orderded) {

        DoubleTensor[] array = new DoubleTensor[orderded.size()];
        for (int i = 0; i < orderded.size(); i++) {
            array[i] = lookup.get(orderded.get(i).getId());
        }

        return array;
    }

    private Map<VariableReference, DoubleTensor> toMap(DoubleTensor[] values, List<Vertex<DoubleTensor>> orderded) {

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
        LogProbGradientCalculator gradientCalculator = new LogProbGradientCalculator(
            bayesianNetwork.getLatentOrObservedVertices(),
            bayesianNetwork.getContinuousLatentVertices()
        );
        return optimize(gradientCalculator);
    }

    @Override
    public OptimizedResult maxLikelihood() {
        LogProbGradientCalculator gradientCalculator = new LogProbGradientCalculator(
            bayesianNetwork.getObservedVertices(),
            bayesianNetwork.getContinuousLatentVertices()
        );
        return optimize(gradientCalculator);
    }

    @Override
    public void addFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler) {

    }

    @Override
    public void removeFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler) {

    }

    public static class AdamOptimizerBuilder {
        private BayesianNetwork bayesianNetwork;

        private double alpha = 0.001;
        private double beta1 = 0.9;
        private double beta2 = 0.999;
        private double epsilon = 1e-8;

        AdamOptimizerBuilder() {
        }

        public AdamOptimizerBuilder bayesianNetwork(BayesianNetwork bayesianNetwork) {
            this.bayesianNetwork = bayesianNetwork;
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
            return new AdamOptimizer(bayesianNetwork, alpha, beta1, beta2, epsilon);
        }

        public String toString() {
            return "AdamOptimizer.AdamOptimizerBuilder(bayesianNetwork=" + this.bayesianNetwork + ", alpha=" + this.alpha + ", beta1=" + this.beta1 + ", beta2=" + this.beta2 + ", epsilon=" + this.epsilon + ")";
        }
    }
}
