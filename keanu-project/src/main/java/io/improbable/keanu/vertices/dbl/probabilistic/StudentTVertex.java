package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.StudentT;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;
import java.util.Random;

public class StudentTVertex extends ProbabilisticDouble {
	
	private final DoubleVertex v;
	private final Random random;
	
	public StudentTVertex(DoubleVertex v, Random random) {
		this.v = v;
		this.random = random;
		setValue(sample());
		setParents(v);
	}
	
	public StudentTVertex(DoubleVertex v) { this(v, new Random()); }
	
	public StudentTVertex(double v, Random random) {
		this(new ConstantDoubleVertex(v), random);
	}
	
	public StudentTVertex(double v) {
		this(new ConstantDoubleVertex(v), new Random());
	}
	
	public DoubleVertex getV() { return v; }
	
	public double density(Double value) { return StudentT.pdf(v.getValue(), value); }
	
	@Override
	public double logPdf(Double value) { return StudentT.logPdf(v.getValue(), value); }
	
	public Map<String, Double> dDensityAtValue() {
		StudentT.Diff diff = StudentT.dPdf(v.getValue(), getValue());
		return convertDualNumbersToDiff(diff.dPdv, diff.dPdt);
	}
	
	@Override
	public Map<String, Double> dLogPdf(Double value) {
		StudentT.Diff diff = StudentT.dlnPdf(v.getValue(), getValue());
		return convertDualNumbersToDiff(diff.dPdv, diff.dPdt);
	}
	
	private Map<String, Double> convertDualNumbersToDiff(double dPdv, double dPdt) {
		PartialDerivatives dPdInputsFromV = v.getDualNumber().getPartialDerivatives().multiplyBy(dPdv);
		PartialDerivatives dPdInputs = dPdInputsFromV;
		dPdInputs.asMap().put(getId(), dPdt);
		
		return dPdInputs.asMap();
	}
	
	@Override
	public Double sample() { return StudentT.sample(v.getValue(), random); }
}
