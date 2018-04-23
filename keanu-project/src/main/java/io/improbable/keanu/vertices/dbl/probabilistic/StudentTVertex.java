package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.StudentT;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.intgr.DoubleVertex;

import java.util.Map;
import java.util.Random;

public class StudentTVertex extends ProbabilisticDouble {
	
	private final DoubleVertex v;
	private final DoubleVertex mu;
	private final DoubleVertex sigma;
	private final Random random;
	
	public StudentTVertex(DoubleVertex v, DoubleVertex mu, DoubleVertex sigma, Random random) {
		this.v = v;
		this.mu = mu;
		this.sigma = sigma;
		this.random = random;
		setValue(sample());
		setParents(v, mu, sigma);
	}
	
	public StudentTVertex(DoubleVertex v, Random random) {
		this(v, 0., 1., random);
	}
	
	public StudentTVertex(DoubleVertex v) { this(v, new Random()); }
	
	public StudentTVertex(DoubleVertex v, DoubleVertex mu, DoubleVertex sigma) { this(v, mu, sigma, new Random()); }
	
	public StudentTVertex(double v, Random random) {
		this(new ConstantDoubleVertex(v), random);
	}
	
	public StudentTVertex(double v) {
		this(new ConstantDoubleVertex(v), new Random());
	}
	
	public StudentTVertex(double v, double mu, double sigma, Random random) {
		this(new ConstantDoubleVertex(v), new ConstantDoubleVertex(mu), new ConstantDoubleVertex(sigma), random);
	}
	
	public StudentTVertex(double vdouble mu, double sigma, ) {
		this(new ConstantDoubleVertex(v), new ConstantDoubleVertex(mu), new ConstantDoubleVertex(sigma), new Random());
	}
	
	public DoubleVertex getV() { return v; }
	
	public DoubleVertex getMu() { return mu; }
	
	public DoubleVertex getSigma() { return sigma; }
	
	@Override
	public double density(Double value) { return StudentT.pdf(v.getValue(), mu.getValue(), sigma.getValue(), value); }
	
	@Override
	public Double sample() { return StudentT.sample(v.getValue(), mu.getValue(), sigma.getValue(), random); }
}
