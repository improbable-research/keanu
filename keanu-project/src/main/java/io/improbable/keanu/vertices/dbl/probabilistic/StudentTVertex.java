package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.StudentT;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.intgr.DoubleVertex;

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

	public StudentTVertex(int v, Random random) {
		this(new ConstantDoubleVertex(v), random);
	}

	public StudentTVertex(int v) {
		this(new ConstantDoubleVertex(v), new Random());
	}
	
	public DoubleVertex getV() { return v; }
	
	@Override
	public double density(Double value) { return StudentT.pdf(v.getValue(), value); }
	
	@Override
	public Double sample() { return StudentT.sample(v.getValue(), random); }
}
