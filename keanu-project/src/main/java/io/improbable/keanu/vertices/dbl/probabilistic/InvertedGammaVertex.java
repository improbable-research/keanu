package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.InvertedGamma;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

import java.util.Map;
import java.util.Random;

public class InvertedGammaVertex extends ProbabilisticDouble {

    private DoubleVertex a;
    private DoubleVertex b;
    private Random random;


    public InvertedGammaVertex(DoubleVertex a, DoubleVertex b, Random random) {
        this.a = a;
        this.b = b;
        this.random = random;
        setParents(a, b);
        setValue(sample());
    }

    public InvertedGammaVertex(DoubleVertex a, DoubleVertex b) {
        this(a, b, new Random());
    }

    public InvertedGammaVertex(double a, double b, Random random) {
        this(new ConstantDoubleVertex(a), new ConstantDoubleVertex(b), random);
    }

    public InvertedGammaVertex(DoubleVertex a, double b, Random random) {
        this(a, new ConstantDoubleVertex(b), random);
    }

    public InvertedGammaVertex(double a, DoubleVertex b, Random random) {
        this(new ConstantDoubleVertex(a), b, random);
    }

    @Override
    public Double sample() {
        return InvertedGamma.sample(a.getValue(), b.getValue(), random);
    }

    @Override
    public double density(Double value) {
        return InvertedGamma.pdf(a.getValue(), b.getValue(), value);
    }

    public double logDensity(Double value) {
        return InvertedGamma.logPdf(a.getValue(), b.getValue(), value);
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        InvertedGamma.Diff dP = InvertedGamma.dPdf(a.getValue(), b.getValue(), getValue());
        return convertDualNumbersToDiff(dP.dPda, dP.dPdb, dP.dPdx);
    }

    public Map<String, Double> dlnDensityAtValue() {
        InvertedGamma.Diff dP = InvertedGamma.dlnPdf(a.getValue(), b.getValue(), getValue());
        return convertDualNumbersToDiff(dP.dPda, dP.dPdb, dP.dPdx);
    }

    private Map<String, Double> convertDualNumbersToDiff(double dPda, double dPdb, double dPdx) {
        Infinitesimal dPdInputsFromA = a.getDualNumber().getInfinitesimal().multiplyBy(dPda);
        Infinitesimal dPdInputsFromB = b.getDualNumber().getInfinitesimal().multiplyBy(dPdb);
        Infinitesimal dPdInputs = dPdInputsFromA.add(dPdInputsFromB);

        dPdInputs.getInfinitesimals().put(getId(), dPdx);

        return dPdInputs.getInfinitesimals();
    }

}
