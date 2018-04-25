package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

import java.util.Map;
import java.util.Random;

import static org.apache.commons.math3.special.Erf.erf;

public class FuzzyCastToIntegerVertex extends ProbabilisticInteger {

    private DoubleVertex input;
    private DoubleVertex fuzzinessSigma;
    private Vertex<Integer> min;
    private Vertex<Integer> max;
    private Random random;

    /**
     * Takes a double and casts it to an integer with a user definable level of fuzziness over the value cast to. The range
     * of potential integer values cast to is specified with a min and max (inclusive). The probability of casting to a
     * given integer is represented as a Gaussian distribution centred on the input value, with a use specifiable sigma.
     * E.n., a sigma value of 0 will guarantee casting ot the nearest integer value with half up rounding.
     *
     * @param input
     * @param fuzzinessSigma fuzziness is represented as a Gaussian distribution with mu of the input value and this sigma.
     * @param min            inclusive
     * @param max            inclusive
     * @param random
     */
    public FuzzyCastToIntegerVertex(DoubleVertex input, DoubleVertex fuzzinessSigma,
                                    Vertex<Integer> min, Vertex<Integer> max, Random random) {

        this.input = input;
        this.fuzzinessSigma = fuzzinessSigma;
        this.min = min;
        this.max = max;
        this.random = random;
        setParents(input, fuzzinessSigma, min, max);
    }

    public FuzzyCastToIntegerVertex(DoubleVertex input, double fuzzinessSigma, int min, int max, Random random) {
        this(input, new ConstantDoubleVertex(fuzzinessSigma), new ConstantIntegerVertex(min),
                new ConstantIntegerVertex(max), random);
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

    @Override
    public double density(Integer value) {
        double i = value;
        double x = getClampedInput();
        double sigma = fuzzinessSigma.getValue();
        double positiveErrorFunction = s(i + 0.5 - x, sigma);
        double negativeErrorFunction = s(i - 0.5 - x, sigma);

        return lambda(x, sigma) * (positiveErrorFunction - negativeErrorFunction);
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        int i = getValue();
        double x = input.getValue();
        double clampedX = getClampedInput();
        double sigma = fuzzinessSigma.getValue();

        double dPdInput = clampedX == x ? dPdx(x, i, sigma) : 0.0;
        double dPdSigma = dPdSigma(clampedX, i, sigma);

        return convertDualNumbersToDiff(dPdInput, dPdSigma);
    }

    @Override
    public Map<String, Double> dlnDensityAtValue() {
        int i = getValue();
        double x = input.getValue();
        double clampedX = getClampedInput();
        double sigma = fuzzinessSigma.getValue();

        double dPdInput = clampedX == x ? dPdx(x, i, sigma) : 0.0;
        double dPdSigma = dPdSigma(clampedX, i, sigma);

        double p = densityAtValue();
        double dlnPdInput = dPdInput / p;
        double dlnPdSigma = dPdSigma / p;

        return convertDualNumbersToDiff(dlnPdInput, dlnPdSigma);
    }

    @Override
    public Integer sample() {
        double fuzzyDouble = sampleFuzzyDoubleInBounds();
        return (int) Math.round(fuzzyDouble);
    }

    private double sampleFuzzyDoubleInBounds() {
        double mu = getClampedInput();
        double sigma = fuzzinessSigma.getValue();
        int min = this.min.getValue();
        int max = this.max.getValue();

        double doubleInBounds;

        do {
            doubleInBounds = Gaussian.sample(mu, sigma, random);
        } while (doubleInBounds < (min - 0.5) || doubleInBounds > (max + 0.5 - 1e-10));

        return doubleInBounds;
    }

    private double getClampedInput() {
        double sigma = fuzzinessSigma.getValue();
        double min = this.min.getValue() - sigma;
        double max = this.max.getValue() + sigma;
        return Math.min(Math.max(input.getValue(), min), max);
    }

    private Map<String, Double> convertDualNumbersToDiff(double dPdInput, double dPdSigma) {
        Infinitesimal dPdInputsFromInput = input.getDualNumber().getInfinitesimal().multiplyBy(dPdInput);
        Infinitesimal dPdInputsFromSigma = fuzzinessSigma.getDualNumber().getInfinitesimal().multiplyBy(dPdSigma);
        Infinitesimal dPdInputs = dPdInputsFromInput.add(dPdInputsFromSigma);

        return dPdInputs.getInfinitesimals();
    }

    private double s(double x, double sigma) {
        return (erf(x / (Math.sqrt(2) * sigma)) / 2.0) + 0.5;
    }

    private double lambda(double x, double sigma) {
        int max = this.max.getValue();
        int min = this.min.getValue();
        return 1.0 / (s(max + 0.5 - x, sigma) - s(min - 0.5 - x, sigma));
    }

    private double dPdx(double x, int i, double sigma) {
        int max = this.max.getValue();
        int min = this.min.getValue();
        double p = density(i);
        return -lambda(x, sigma) * (N(i + 0.5 - x, sigma) - N(i - 0.5 - x, sigma)
                - p * (N(max + 0.5 - x, sigma) - N(min - 0.5 - x, sigma)));
    }

    private double dPdSigma(double x, int i, double sigma) {
        int max = this.max.getValue();
        int min = this.min.getValue();
        double p = density(i);
        return lambda(x, sigma) * (dSdSigma(i + 0.5 - x, sigma) - dSdSigma(i - 0.5 - x, sigma)
                - p * (dSdSigma(max + 0.5 - x, sigma) - dSdSigma(min - 0.5 - x, sigma)));
    }

    private double dSdSigma(double x, double sigma) {
        return (-x * N(x, sigma)) / sigma;
    }

    private double N(double mu, double sigma) {
        return n(mu, sigma) / Math.sqrt(2 * Math.PI * sigma * sigma);
    }

    private double n(double mu, double sigma) {
        return Math.exp(-(mu * mu) / (2.0 * sigma * sigma));
    }
}
