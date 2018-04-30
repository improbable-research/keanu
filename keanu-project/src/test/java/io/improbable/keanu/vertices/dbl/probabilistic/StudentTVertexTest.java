package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.apache.commons.math3.distribution.TDistribution;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public class StudentTVertexTest {
	
	private static final double DELTA = 0.0001;
	
	private final Logger log = LoggerFactory.getLogger(StudentTVertexTest.class);
	
	private Random random;
	private double mu = 0.;
	private double sigma = 1.;
	
	@Before
	public void setup() {
		random = new Random(1);
	}
	
	// http://commons.apache.org/proper/commons-math/javadocs/api-3.6/org/apache/commons/math3/distribution/TDistribution.html
	// log.info("");
	
	@Test
	public void studentDensityMethod() {
		double v = 1.;
		double x = 1.;
		TDistribution t = new TDistribution();
		System.out.println(t.getNumericalVariance());
		//double density = density(x);
	}
	
	/*
	@Test
	public void studentSampleMethodMatchesDensityMethod() {
		assert false; // TODO: implement
	}
	
	@Test
	public void samplingMatchesPdf() {
		assert false; // TODO: implement
	}
	
	@Test
	public void logDensityIsSameAsLogOfDensity() {
		assert false; // TODO: implement
	}
	
	@Test
	public void diffLnDensityIsSameAsLogOfDiffDensity() {
		assert false; // TODO: implement
	}
	
	
	@Test
	public void inferHyperParamsFromSamples() {
		assert false; // TODO: implement
	}
	*/
}
