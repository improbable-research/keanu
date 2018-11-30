package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.Builder;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * https://arxiv.org/pdf/1412.6980.pdf
 */
@Builder
public class AdamOptimizer implements Optimizer {

    @Getter
    private final BayesianNetwork bayesianNetwork;

    private final double alpha;
    private final double beta1;
    private final double beta2;

    private void optimize(Function<BayesianNetwork, List<DoubleTensor>> gradientCalculator) {

        List<DoubleTensor> theta = getTheta(bayesianNetwork);
        List<DoubleTensor> m = getZeros(theta);
        List<DoubleTensor> v = getZeros(theta);

        double eps = 1e-8;
        double alpha = 0.001;
        int t = 0;

        while (t == 0 || !hasConverged()) {
            t++;
            List<DoubleTensor> gradientT = gradientCalculator.apply(bayesianNetwork);
            List<DoubleTensor> mT = momentumUpdate(beta1, m, gradientT);
            List<DoubleTensor> vT = momentumUpdate(beta2, v, squared(gradientT));

            List<DoubleTensor> mHat = div(mT, (1 - Math.pow(beta1, t)));
            List<DoubleTensor> vHat = div(vT, (1 - Math.pow(beta2, t)));

            List<DoubleTensor> sqrtVHatPlusEps = add(sqrt(vHat), eps);
            List<DoubleTensor> alphaTimesMHatDivVHatPlusEps = mul(div(mHat, sqrtVHatPlusEps), alpha);

            theta = sub(theta, alphaTimesMHatDivVHatPlusEps);
        }
    }

    private List<DoubleTensor> getTheta(BayesianNetwork bayesianNetwork) {
        return bayesianNetwork.getLatentVertices().stream()
            .map(v -> (DoubleTensor) v.getValue())
            .collect(Collectors.toList());
    }

    private List<DoubleTensor> getZeros(List<DoubleTensor> values) {
        return values.stream()
            .map(v -> DoubleTensor.zeros(v.getShape()))
            .collect(Collectors.toList());
    }

    private List<DoubleTensor> momentumUpdate(double beta, List<DoubleTensor> momentum, List<DoubleTensor> gradient) {

        List<DoubleTensor> update = new ArrayList<>();
        for (int i = 0; i < momentum.size(); i++) {
            update.add(momentum.get(i).times(beta).plus(gradient.get(i).times(1 - beta)));
        }

        return update;
    }

    private List<DoubleTensor> squared(List<DoubleTensor> values) {
        return values.stream().map(v -> v.pow(2)).collect(Collectors.toList());
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

    private List<DoubleTensor> sub(List<DoubleTensor> left, List<DoubleTensor> right) {
        List<DoubleTensor> result = new ArrayList<>();
        for (int i = 0; i < left.size(); i++) {
            result.add(left.get(i).minus(right.get(i)));
        }
        return result;
    }

    private List<DoubleTensor> mul(List<DoubleTensor> values, double factor) {
        return values.stream().map(v -> v.times(factor)).collect(Collectors.toList());
    }

    private List<DoubleTensor> add(List<DoubleTensor> values, double addition) {
        return values.stream().map(v -> v.plus(addition)).collect(Collectors.toList());
    }

    private boolean hasConverged() {
        return true;
    }

    @Override
    public double maxAPosteriori() {
        return 0;
    }

    @Override
    public double maxLikelihood() {
        return 0;
    }

    @Override
    public void addFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler) {

    }

    @Override
    public void removeFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler) {

    }
}
