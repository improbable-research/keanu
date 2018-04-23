package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.StudentT;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.intgr.DoubleVertex;

import java.util.Map;
import java.util.Random;

public class StudentTVertex extends ProbabilisticDouble {
	
	private final DoubleVertex t;
	private final DoubleVertex v;
	private final Random random;
	
	public StudentTVertex(DoubleVertex t, DoubleVertex v, Random random) {
		this.t = t;
		this.v = v;
		this.random = random;
		setValue(sample());
		setParents(t, v);
	}
	
	public StudentTVertex(DoubleVertex t, DoubleVertex v) { this(t, v, new Random()); }

	public StudentTVertex(int t, int v, Random random) {
		this(new ConstantDoubleVertex(t), new ConstantDoubleVertex(v), random);
	}

	public StudentTVertex(int t, int v) {
		this(new ConstantDoubleVertex(t), new ConstantDoubleVertex(v), new Random());
	}
	
	public DoubleVertex getT() { return t; }
	
	public DoubleVertex getV() { return v; }
	
	@Override
	public double density(Double value) { return StudentT.pdf(t.getValue(), v.getValue(), value); }
}
