package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Logistic;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

import java.util.Map;
import java.util.Random;

public class LogisticVertex extends ProbabilisticDouble {

    private final DoubleVertex a;
    private final DoubleVertex b;
    private final Random random;

    public LogisticVertex(DoubleVertex a, DoubleVertex b, Random random) {
        this.a = a;
        this.b = b;
        this.random = random;
        setParents(a, b);
    }

    public LogisticVertex(DoubleVertex a, double b, Random random) {
        this(a, new ConstantDoubleVertex(b), random);
    }

    public LogisticVertex(double a, DoubleVertex b, Random random) {
        this(new ConstantDoubleVertex(a), b, random);
    }

    public LogisticVertex(double a, double b, Random random) {
        this(new ConstantDoubleVertex(a), new ConstantDoubleVertex(b), random);
    }

    public LogisticVertex(DoubleVertex a, DoubleVertex b) {
        this(a, b, new Random());
    }

    public LogisticVertex(DoubleVertex a, double b) {
        this(a, new ConstantDoubleVertex(b), new Random());
    }

    public LogisticVertex(double a, DoubleVertex b) {
        this(new ConstantDoubleVertex(a), b, new Random());
    }

    public LogisticVertex(double a, double b) {
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
        return Logistic.pdf(a.getValue(), b.getValue(), value);
    }

    public double logDensity(Double value) {
        return Logistic.logPdf(a.getValue(), b.getValue(), value);
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        Logistic.Diff diff = Logistic.dPdf(a.getValue(), b.getValue(), getValue());
        return convertDualNumbersToDiff(diff.dPda, diff.dPdb, diff.dPdx);
    }

    @Override
    public Map<String, Double> dlnDensityAtValue() {
        Logistic.Diff diff = Logistic.dlnPdf(a.getValue(), b.getValue(), getValue());
        return convertDualNumbersToDiff(diff.dPda, diff.dPdb, diff.dPdx);
    }

    private Map<String, Double> convertDualNumbersToDiff(double dPda, double dPdb, double dPdx) {
        Infinitesimal dPdInputsFromMu = a.getDualNumber().getInfinitesimal().multiplyBy(dPda);
        Infinitesimal dPdInputsFromSigma = b.getDualNumber().getInfinitesimal().multiplyBy(dPdb);
        Infinitesimal dPdInputs = dPdInputsFromMu.add(dPdInputsFromSigma);

        dPdInputs.getInfinitesimals().put(getId(), dPdx);
        return dPdInputs.getInfinitesimals();
    }

    @Override
    public Double sample() {
        return Logistic.sample(a.getValue(), b.getValue(), random);
    }
}
