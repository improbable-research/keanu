package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import lombok.Builder;

import java.util.ArrayList;
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
        List<DoubleTensor> m = getZeros(theta);
        List<DoubleTensor> v = getZeros(theta);

        int t = 0;
        boolean converged = false;

        int maxIterations = Integer.MAX_VALUE;

        while (!converged && t < maxIterations) {
            t++;

            setTheta(theta, latentVariables);
            List<DoubleTensor> gradientT = toList(gradientCalculator.getJointLogProbGradientWrtLatents(), latentVariables);

            m = updateMomentEstimate(beta1, m, gradientT);
            v = updateMomentEstimate(beta2, v, squared(gradientT));

            List<DoubleTensor> mHat = div(m, (1 - Math.pow(beta1, t)));
            List<DoubleTensor> vHat = div(v, (1 - Math.pow(beta2, t)));

            List<DoubleTensor> sqrtVHatPlusEps = add(sqrt(vHat), epsilon);
            List<DoubleTensor> alphaTimesMHatDivVHatPlusEps = mul(div(mHat, sqrtVHatPlusEps), alpha);

            List<DoubleTensor> thetaNext = add(theta, alphaTimesMHatDivVHatPlusEps);

            converged = hasConverged(gradientT, thetaNext, theta);

//            print(gradientT);
//            print(thetaNext);
            theta = thetaNext;
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

    private List<DoubleTensor> updateMomentEstimate(double beta, List<DoubleTensor> moment, List<DoubleTensor> gradient) {

        List<DoubleTensor> update = new ArrayList<>();
        for (int i = 0; i < moment.size(); i++) {
            update.add(moment.get(i).times(beta).plus(gradient.get(i).times(1 - beta)));
        }

        return update;
    }

    private List<DoubleTensor> toList(Map<VertexId, DoubleTensor> gradients, List<Vertex<DoubleTensor>> latentOrder) {
        return latentOrder.stream().map(v -> gradients.get(v.getId())).collect(Collectors.toList());
    }

    private void print(List<DoubleTensor> values) {
        values.forEach(v -> System.out.println(Arrays.toString(v.asFlatDoubleArray())));
    }

    private List<DoubleTensor> squared(List<DoubleTensor> values) {
        return values.stream().map(v -> v.pow(2)).collect(Collectors.toList());
    }

    private double sum(List<DoubleTensor> values) {
        return values.stream().mapToDouble(NumberTensor::sum).sum();
    }

    private List<DoubleTensor> sqrt(List<DoubleTensor> values) {
        return values.stream().map(DoubleTensor::sqrt).collect(Collectors.toList());
    }

    private List<DoubleTensor> div(List<DoubleTensor> values, double divisor) {
        return values.stream().map(v -> v.div(divisor)).collect(Collectors.toList());
    }

    private List<DoubleTensor> div(List<DoubleTensor> numerators, List<DoubleTensor> denominators) {
        List<DoubleTensor> result = new ArrayList<>();
        for (int i = 0; i < numerators.size(); i++) {
            result.add(numerators.get(i).div(denominators.get(i)));
        }
        return result;
    }

    private List<DoubleTensor> add(List<DoubleTensor> left, List<DoubleTensor> right) {
        List<DoubleTensor> result = new ArrayList<>();
        for (int i = 0; i < left.size(); i++) {
            result.add(left.get(i).plus(right.get(i)));
        }
        return result;
    }

    private List<DoubleTensor> mul(List<DoubleTensor> values, double factor) {
        return values.stream().map(v -> v.times(factor)).collect(Collectors.toList());
    }

    private List<DoubleTensor> add(List<DoubleTensor> values, double addition) {
        return values.stream().map(v -> v.plus(addition)).collect(Collectors.toList());
    }

    private boolean hasConverged(List<DoubleTensor> gradient, List<DoubleTensor> thetaPrevious, List<DoubleTensor> theta) {

//        double thetaDeltaMag = Math.sqrt(sum(squared(sub(theta, thetaPrevious))));
//
//        if (thetaDeltaMag < 1e-6) {
//            return true;
//        }

        double gradientMag = Math.sqrt(sum(squared(gradient)));

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
