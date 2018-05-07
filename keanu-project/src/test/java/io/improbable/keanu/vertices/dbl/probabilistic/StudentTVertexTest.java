package io.improbable.keanu.vertices.dbl.probabilistic;

import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static junit.framework.TestCase.assertEquals;

public class StudentTVertexTest {
	
	private static final double DELTA = 0.0001;
	
	private static int POS_V = 0;
	private static int POS_MEAN = 1;
	private static int POS_SD = 2; // standard deviation
	private static final double[] TEST_VALUES = new double[]{
			1.0, 2.0, 3.0
	};
	
	private final Logger log = LoggerFactory.getLogger(StudentTVertexTest.class);
	
	private Random random;
	private double mu = 0.;
	private double sigma = 1.;
	
	@Before
	public void setup() {
		random = new Random(1);
	}
	
	@Test
	public void sampleTest() {
		int N = 10000000;
		double[][] test_values = {
				// v, mean, sd
				{2., 0., 4.5}
		};
		for (int i = 0; i < test_values.length; i++) {
			double v = test_values[i][POS_V];
			StudentTVertex studentT = new StudentTVertex(v, random);
			
			List<Double> samples = new ArrayList<>();
			for (int j = 0; j < N; j++) {
				double sample = studentT.sample();
				samples.add(sample);
			}
			
			SummaryStatistics stats = new SummaryStatistics();
			samples.forEach(stats::addValue);
			
			double mean = stats.getMean();
			double sd = stats.getStandardDeviation();
			log.trace("Degrees of freedom: " + v);
			log.trace("Mean: " + mean);
			log.trace("Standard deviation: " + sd);
			Assert.assertEquals(test_values[i][POS_MEAN], mean, 0.1);
			Assert.assertEquals(test_values[i][POS_SD], sd, 0.1);
		}
	}
	
	@Test
	public void pdfTest() {
		for (int i = 0; i < TEST_VALUES.length; i++) {
			testPdfAtGivenDegreesOfFreedom(TEST_VALUES[i]);
		}
	}
	
	private void testPdfAtGivenDegreesOfFreedom(double v) {
		TDistribution apache = new TDistribution(v);
		StudentTVertex studentT = new StudentTVertex(v, random);
		
		for(double t = -4.5; t <= 4.5; t += 0.5) {
			double expected = apache.density(t);
			double actual = studentT.density(t);
			assertEquals(expected, actual, DELTA);
		}
	}
}
