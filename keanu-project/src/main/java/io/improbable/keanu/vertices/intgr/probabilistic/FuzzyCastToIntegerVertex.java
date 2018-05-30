package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

import java.util.Map;

import static org.apache.commons.math3.special.Erf.erf;

/**
 * Takes a double and casts it to an integer with a user definable level of fuzziness over the value cast to. The range
 * of potential integer values cast to is specified with a min and max (inclusive). The probability of casting to a
 * given integer is represented as a Gaussian distribution centred on the input value, with a user specifiable sigma.
 * e.g. a sigma value of 0 will guarantee casting to the nearest integer value with half up rounding.
 */
public class FuzzyCastToIntegerVertex extends ProbabilisticInteger {

    private DoubleVertex input;
    private DoubleVertex fuzzinessSigma;
    private Vertex<Integer> min;
    private Vertex<Integer> max;

    /**
     * @param input          vertex intended for casting
     * @param fuzzinessSigma fuzziness is represented as a Gaussian distribution with mu of the input value and this sigma.
     * @param min            inclusive
     * @param max            inclusive
     */
    public FuzzyCastToIntegerVertex(DoubleVertex input,
                                    DoubleVertex fuzzinessSigma,
                                    Vertex<Integer> min,
                                    Vertex<Integer> max) {

        this.input = input;
        this.fuzzinessSigma = fuzzinessSigma;
        this.min = min;
        this.max = max;
        setParents(input, fuzzinessSigma, min, max);
    }

    public FuzzyCastToIntegerVertex(DoubleVertex input,
                                    double fuzzinessSigma,
                                    int min,
                                    int max) {
        this(input,
            new ConstantDoubleVertex(fuzzinessSigma),
            new ConstantIntegerVertex(min),
            new ConstantIntegerVertex(max)
        );
    }

    public DoubleVertex getInput() {
        return input;
    }

    public DoubleVertex getFuzzinessSigma() {
        return fuzzinessSigma;
    }

    public Vertex<Integer> getMin() {
        return min;
    }

    public Vertex<Integer> getMax() {
        return max;
    }

    private double density(Integer value) {
        double i = value;
        double x = getClampedInput();
        double sigma = fuzzinessSigma.getValue();
        double positiveErrorFunction = s(i + 0.5 - x, sigma);
        double negativeErrorFunction = s(i - 0.5 - x, sigma);

        return lambda(x, sigma) * (positiveErrorFunction - negativeErrorFunction);
    }

    @Override
    public double logPmf(Integer value) {
        return Math.log(density(value));
    }

    @Override
    public Map<Long, DoubleTensor> dLogPmf(Integer value) {
        int i = getValue();
        double x = input.getValue();
        double clampedX = getClampedInput();
        double sigma = fuzzinessSigma.getValue();

        double dPdInput = clampedX == x ? dPdx(x, i, sigma) : 0.0;
        double dPdSigma = dPdSigma(clampedX, i, sigma);

        double p = density(value);
        double dlnPdInput = dPdInput / p;
        double dlnPdSigma = dPdSigma / p;

        return convertDualNumbersToDiff(dlnPdInput, dlnPdSigma);
    }

    @Override
    public Integer sample(KeanuRandom random) {
        double fuzzyDouble = sampleFuzzyDoubleInBounds(random);
        return (int) Math.round(fuzzyDouble);
    }

    private double sampleFuzzyDoubleInBounds(KeanuRandom random) {
        double mu = getClampedInput();
        double sigma = fuzzinessSigma.getValue();

        double doubleInBounds;

        do {
            doubleInBounds = Gaussian.sample(mu, sigma, random);
        } while (doubleInBounds < (min.getValue() - 0.5) || doubleInBounds > (max.getValue() + 0.5 - 1e-10));

        return doubleInBounds;
    }

    private double getClampedInput() {
        double sigma = fuzzinessSigma.getValue();
        double minClamped = this.min.getValue() - sigma;
        double maxClamped = this.max.getValue() + sigma;
        return Math.min(Math.max(input.getValue(), minClamped), maxClamped);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(double dPdInput, double dPdSigma) {
        PartialDerivatives dPdInputsFromInput = input.getDualNumber().getPartialDerivatives().multiplyBy(dPdInput);
        PartialDerivatives dPdInputsFromSigma = fuzzinessSigma.getDualNumber().getPartialDerivatives().multiplyBy(dPdSigma);
        PartialDerivatives dPdInputs = dPdInputsFromInput.add(dPdInputsFromSigma);

        return DoubleTensor.fromScalars(dPdInputs.asMap());
    }

    private double s(double x, double sigma) {
        return (erf(x / (Math.sqrt(2) * sigma)) / 2.0) + 0.5;
    }

    private double lambda(double x, double sigma) {
        return 1.0 / (s(max.getValue() + 0.5 - x, sigma) - s(min.getValue() - 0.5 - x, sigma));
    }

    private double dPdx(double x, int i, double sigma) {
        double p = density(i);
        return -lambda(x, sigma) * (n(i + 0.5 - x, sigma) - n(i - 0.5 - x, sigma)
            - p * (n(max.getValue() + 0.5 - x, sigma) - n(min.getValue() - 0.5 - x, sigma)));
    }

    private double dPdSigma(double x, int i, double sigma) {
        double p = density(i);
        return lambda(x, sigma) * (dSdSigma(i + 0.5 - x, sigma) - dSdSigma(i - 0.5 - x, sigma)
            - p * (dSdSigma(max.getValue() + 0.5 - x, sigma) - dSdSigma(min.getValue() - 0.5 - x, sigma)));
    }

    private double dSdSigma(double x, double sigma) {
        return (-x * n(x, sigma)) / sigma;
    }

    private double n(double mu, double sigma) {
        return Gaussian.pdf(mu, sigma, 0);
    }
}
