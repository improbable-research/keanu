package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Exponential;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

import java.util.Map;
import java.util.Random;

public class ExponentialVertex extends ProbabilisticDouble {

    private final DoubleVertex a;
    private final DoubleVertex b;
    private final Random random;

    public ExponentialVertex(DoubleVertex a, DoubleVertex b, Random random) {
        this.a = a;
        this.b = b;
        this.random = random;
        setParents(a, b);
    }

    public ExponentialVertex(DoubleVertex a, double b, Random random) {
        this(a, new ConstantDoubleVertex(b), random);
    }

    public ExponentialVertex(double a, DoubleVertex b, Random random) {
        this(new ConstantDoubleVertex(a), b, random);
    }

    public ExponentialVertex(double a, double b, Random random) {
        this(new ConstantDoubleVertex(a), new ConstantDoubleVertex(b), random);
    }

    public ExponentialVertex(DoubleVertex a, DoubleVertex b) {
        this(a, b, new Random());
    }

    public ExponentialVertex(DoubleVertex a, double b) {
        this(a, new ConstantDoubleVertex(b), new Random());
    }

    public ExponentialVertex(double a, DoubleVertex b) {
        this(new ConstantDoubleVertex(a), b, new Random());
    }

    public ExponentialVertex(double a, double b) {
        this(new ConstantDoubleVertex(a), new ConstantDoubleVertex(b), new Random());
    }

    public DoubleVertex getA() {
        return a;
    }

    public DoubleVertex getB() {
        return b;
    }

    @Override
    public double density(Double value) {
        return Exponential.pdf(a.getValue(), b.getValue(), value);
    }

    public double logDensity(Double value) {
        return Exponential.logPdf(a.getValue(), b.getValue(), value);
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        Exponential.Diff dP = Exponential.dPdf(a.getValue(), b.getValue(), getValue());
        return convertDualNumbersToDiff(dP.dPda, dP.dPdb, dP.dPdx);
    }

    @Override
    public Map<String, Double> dlnDensityAtValue() {
        Exponential.Diff dP = Exponential.dlnPdf(a.getValue(), b.getValue(), getValue());
        return convertDualNumbersToDiff(dP.dPda, dP.dPdb, dP.dPdx);
    }

    @Override
    public Double sample() {
        return Exponential.sample(a.getValue(), b.getValue(), random);
    }

    private Map<String, Double> convertDualNumbersToDiff(double dPda, double dPdb, double dPdx) {
        Infinitesimal dPdInputsFromMu = a.getDualNumber().getInfinitesimal().multiplyBy(dPda);
        Infinitesimal dPdInputsFromSigma = b.getDualNumber().getInfinitesimal().multiplyBy(dPdb);
        Infinitesimal dPdInputs = dPdInputsFromMu.add(dPdInputsFromSigma);

        dPdInputs.getInfinitesimals().put(getId(), dPdx);
        return dPdInputs.getInfinitesimals();
    }

}
