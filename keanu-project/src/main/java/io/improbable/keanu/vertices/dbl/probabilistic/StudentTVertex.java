package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.StudentT;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 *
 */
public class StudentTVertex extends ProbabilisticDouble {
	private final IntegerVertex v;
	private final Random random;
	/**
	 *
	 * @param v Degrees of Freedom
	 * @param random random number generator (RNG) seed
	 */
	public StudentTVertex(IntegerVertex v, Random random) {
		this.v = v;
		this.random = random;
		setParents(v);
	}
	/**
	 *
	 * @param v Degrees of Freedom
	 */
	public StudentTVertex(IntegerVertex v) { this(v, new Random()); }
	/**
	 *
	 * @param v Degrees of Freedom
	 * @param random random number generator (RNG) seed
	 */
	public StudentTVertex(int v, Random random) {
		this(new ConstantIntegerVertex(v), random);
	}
	/**
	 *
	 * @param v Degrees of Freedom
	 */
	public StudentTVertex(int v) {
		this(new ConstantIntegerVertex(v), new Random());
	}
	/**
	 *
	 * @return degrees of freedom (v)
	 */
	public IntegerVertex getV() { return v; }
	/**
	 *
	 * @param t random variable
	 * @return Probability Density of t
	 */
	public double density(Double t) { return StudentT.pdf(v.getValue(), t); }
	
	/**
	 *
	 * @param t random variable
	 * @return Log of the Probability Density of t
	 */
	@Override
	public double logPdf(Double t) { return StudentT.logPdf(v.getValue(), t); }
	
	/**
	 *
	 * @param t random variable
	 * @return Differential of the Probability Density of t
	 */
	public Map<String, DoubleTensor> dPdf(Double t) {
		StudentT.Diff diff = StudentT.dPdf(v.getValue(), t);
		Map<String, DoubleTensor> m = new HashMap<>();
		m.put(getId(), DoubleTensor.scalar(diff.dPdt));
		return m;
	}
	/**
	 *
	 * @param t random variable
	 * @return Differential of the Log of the Probability Density of t
	 */
	@Override
	public Map<String, DoubleTensor> dLogPdf(Double t) {
		StudentT.Diff diff = StudentT.dLogPdf(v.getValue(), t);
		Map<String, DoubleTensor> m = new HashMap<>();
		m.put(getId(), DoubleTensor.scalar(diff.dPdt));
		return m;
	}
	/**
	 *
	 * @return sample of Student T distribution
	 */
	@Override
	public Double sample() { return StudentT.sample(v.getValue(), random); }
}
