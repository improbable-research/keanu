package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import lombok.Builder;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;

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
        List<DoubleTensor> theta = getTheta(latentVariables);
        List<DoubleTensor> thetaNext = getZeros(theta);
        List<DoubleTensor> m = getZeros(theta);
        List<DoubleTensor> v = getZeros(theta);
        List<DoubleTensor> mHat = getZeros(theta);
        List<DoubleTensor> vHat = getZeros(theta);

        int t = 0;
        boolean converged = false;

        int maxIterations = Integer.MAX_VALUE;

        while (!converged && t < maxIterations) {
            t++;

            setTheta(theta, latentVariables);
            List<DoubleTensor> gradientT = toList(gradientCalculator.getJointLogProbGradientWrtLatents(), latentVariables);

            double beta1T = (1 - Math.pow(beta1, t));
            double beta2T = (1 - Math.pow(beta2, t));

            for (int i = 0; i < gradientT.size(); i++) {

                m.set(i, m.get(i).times(beta1)
                    .plus(gradientT.get(i).times(1 - beta1)));

                v.set(i, v.get(i).times(beta2)
                    .plus(gradientT.get(i).pow(2).times(1 - beta2)));

                mHat.set(i, m.get(i).div(beta1T));
                vHat.set(i, v.get(i).div(beta2T));

                thetaNext.set(i,
                    theta.get(i).plus(mHat.get(i).div(vHat.get(i).sqrt().plus(epsilon)).times(alpha))
                );
            }

            converged = hasConverged(gradientT, thetaNext, theta);

//            print(gradientT);
//            print(thetaNext);

            List<DoubleTensor> temp = theta;
            theta = thetaNext;
            thetaNext = temp;
        }

        return bayesianNetwork.getLogOfMasterP();
    }

    private List<DoubleTensor> getTheta(List<Vertex<DoubleTensor>> latentVertices) {
        return latentVertices.stream()
            .map(Vertex::getValue)
            .collect(Collectors.toList());
    }

    private void setTheta(List<DoubleTensor> theta, List<Vertex<DoubleTensor>> latentVertices) {
        for (int i = 0; i < theta.size(); i++) {
            latentVertices.get(i).setValue(theta.get(i));
        }
        VertexValuePropagation.cascadeUpdate(latentVertices);
    }

    private List<DoubleTensor> getZeros(List<DoubleTensor> values) {
        return values.stream()
            .map(v -> DoubleTensor.zeros(v.getShape()))
            .collect(Collectors.toList());
    }

    private List<DoubleTensor> toList(Map<VertexId, DoubleTensor> gradients, List<Vertex<DoubleTensor>> latentOrder) {
        return latentOrder.stream().map(v -> gradients.get(v.getId())).collect(Collectors.toList());
    }

    private void print(List<DoubleTensor> values) {
        values.forEach(v -> System.out.println(Arrays.toString(v.asFlatDoubleArray())));
    }

    private double magnitude(List<DoubleTensor> values) {

        double magPow2 = 0;
        for (int i = 0; i < values.size(); i++) {
            magPow2 += values.get(i).pow(2).sum();
        }

        return Math.sqrt(magPow2);
    }

    private boolean hasConverged(List<DoubleTensor> gradient, List<DoubleTensor> thetaPrevious, List<DoubleTensor> theta) {

//        double thetaDeltaMag = Math.sqrt(sum(squared(sub(theta, thetaPrevious))));
//
//        if (thetaDeltaMag < 1e-6) {
//            return true;
//        }

        double gradientMag = magnitude(gradient);

        if (gradientMag < 1e-6) {
            return true;
        }

        return false;
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
