package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import lombok.Builder;

import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * https://arxiv.org/pdf/1412.6980.pdf
 */
@Builder
public class AdamOptimizer implements Optimizer {

    private final BayesianNetwork bayesianNetwork;

    private final double alpha;
    private final double beta1;
    private final double beta2;
    private final double epsilon;

    private double optimize(LogProbGradientCalculator gradientCalculator) {

        List<Vertex<DoubleTensor>> latentVariables = bayesianNetwork.getContinuousLatentVertices();
        DoubleTensor[] theta = getTheta(latentVariables);
        DoubleTensor[] thetaNext = getZeros(theta);
        DoubleTensor[] m = getZeros(theta);
        DoubleTensor[] v = getZeros(theta);
        DoubleTensor[] mHat = getZeros(theta);
        DoubleTensor[] vHat = getZeros(theta);

        int t = 0;
        boolean converged = false;

        int maxIterations = Integer.MAX_VALUE;

        while (!converged && t < maxIterations) {
            t++;

            setTheta(theta, latentVariables);
            DoubleTensor[] gradientT = toArray(gradientCalculator.getJointLogProbGradientWrtLatents(), latentVariables);

            double beta1T = (1 - Math.pow(beta1, t));
            double beta2T = (1 - Math.pow(beta2, t));

            for (int i = 0; i < gradientT.length; i++) {

                m[i] = m[i].times(beta1).plus(gradientT[i].times(1 - beta1));
                v[i] = v[i].times(beta2).plus(gradientT[i].pow(2).times(1 - beta2));

                mHat[i] = m[i].div(beta1T);
                vHat[i] = v[i].div(beta2T);

                thetaNext[i] = theta[i].plus(mHat[i].div(vHat[i].sqrt().plus(epsilon)).times(alpha));
            }

            converged = hasConverged(gradientT);

            DoubleTensor[] temp = theta;
            theta = thetaNext;
            thetaNext = temp;
        }

        return bayesianNetwork.getLogOfMasterP();
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
    public double maxAPosteriori() {
        LogProbGradientCalculator gradientCalculator = new LogProbGradientCalculator(
            bayesianNetwork.getLatentOrObservedVertices(),
            bayesianNetwork.getContinuousLatentVertices()
        );
        return optimize(gradientCalculator);
    }

    @Override
    public double maxLikelihood() {
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
}
