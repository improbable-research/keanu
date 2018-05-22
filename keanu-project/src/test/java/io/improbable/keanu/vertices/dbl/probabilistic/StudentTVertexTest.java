package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static java.lang.Math.*;
import java.util.ArrayList;
import java.util.List;

import static junit.framework.TestCase.assertEquals;

/**
 *
 */
public class StudentTVertexTest {	private static final double DELTA = 0.0001;	private static int POS_V = 0;
	private static int POS_MEAN = 1;
	private static int POS_SD = 2; // standard deviation
	private static final int[] TEST_VALUES_OF_V = new int[]{
			1, 2, 3
	};
	private final Logger log = LoggerFactory.getLogger(StudentTVertexTest.class);
	private KeanuRandom random;
	/**
	 *
	 */
	@Before
	public void setup() {
		random = new KeanuRandom(1);
	}
	/**
	 * Test the StudentTVertex -> StudentT -> sample()
	 */
	@Test
	public void sampleTest() {
		int N = 10 * 1000 * 1000;
		double sample_delta = 0.1;
		double[][] test_values = {
				// v, mean, sd
				{2., 0., 4.5}
		};
		for (int i = 0; i < test_values.length; i++) {
			int v = (int) test_values[i][POS_V];
			StudentTVertex studentT = new StudentTVertex(v, random);
			
			List<Double> student_samples = new ArrayList<>();
			for (int j = 0; j < N; j++) {
				student_samples.add(studentT.sample(random));
			}
			testSampleMeanAndStdDeviation(v, test_values[i][POS_MEAN], test_values[i][POS_SD], student_samples, sample_delta);
		}
	}
	/**
	 * Test the StudentT Probability Density Function
	 */
	@Test
	public void pdfTest() {
		for (int i = 0; i < TEST_VALUES_OF_V.length; i++) {
			testPdfAtGivenDegreesOfFreedom(TEST_VALUES_OF_V[i]);
		}
	}
	/**
	 * Test the Log of the StudentT Probability Density Function
	 */
	@Test
	public void logPdfTest() {
		for (int i = 0; i < TEST_VALUES_OF_V.length; i++) {
			testLogPdfAtGivenDegreesOfFreedom(TEST_VALUES_OF_V[i]);
		}
	}
	/**
	 * Test the differential of the StudentT Probability Density Function
	 */
	@Test
	public void dPdfTest() {
		for (int i = 0; i < TEST_VALUES_OF_V.length; i++) {
			testDPdfAtGivenDegreesOfFreedom(TEST_VALUES_OF_V[i]);
		}
	}
	/**
	 * Test the differential of the log of the StudentT Probability Density Function
	 */
	@Test
	public void dLogPdfTest() {
		for (int i = 0; i < TEST_VALUES_OF_V.length; i++) {
			testDLogPdfAtGivenDegreesOfFreedom(TEST_VALUES_OF_V[i]);
		}
	}
	private void testSampleMeanAndStdDeviation(int v, double expected_mean, double expected_sd, List<Double> samples, double delta) {
		SummaryStatistics stats = new SummaryStatistics();
		samples.forEach(stats::addValue);
		
		double mean = stats.getMean();
		double sd = stats.getStandardDeviation();
		log.trace("Degrees of freedom: " + v);
		log.trace("Mean: " + mean);
		log.trace("Standard deviation: " + sd);
		Assert.assertEquals(expected_mean, mean, delta);
		Assert.assertEquals(expected_sd, sd, delta);
	}
	/**
	 *
	 * @param v Degrees of Freedom
	 */
	private void testPdfAtGivenDegreesOfFreedom(int v) {
		TDistribution apache = new TDistribution(v);
		StudentTVertex studentT = new StudentTVertex(v, random);
		
		for(double t = -4.5; t <= 4.5; t += 0.5) {
			double expected = apache.density(t);
			double actual = studentT.density(t);
			assertEquals(expected, actual, DELTA);
		}
	}
	/**
	 *
	 * @param v Degrees of Freedom
	 */
	private void testLogPdfAtGivenDegreesOfFreedom(int v) {
		TDistribution apache = new TDistribution(v);
		StudentTVertex studentT = new StudentTVertex(v, random);
		
		for(double t = -4.5; t <= 4.5; t += 0.5) {
			double expected = apache.logDensity(t);
			double actual = studentT.logPdf(t);
			assertEquals(expected, actual, DELTA);
		}
	}
	/**
	 *
	 * @param v Degrees of Freedom
	 */
	private void testDPdfAtGivenDegreesOfFreedom(int v) {
		StudentTVertex studentT = new StudentTVertex(v, random);
		
		for(double t = -4.5; t <= 4.5; t += 0.5) {
			double expected;
			double actual = studentT.dPdf(t).get(studentT.getId()).scalar();
			switch(v) {
				case 1:
					expected = (-2. * t) / (PI * pow(pow(t, 2) + 1., 2));
					break;
				case 2:
					expected = (-3. * t) / pow(pow(t, 2) + 2., 5. / 2.);
					break;
				case 3:
					expected = (-24. * sqrt(3) * t) / (PI * pow(pow(t, 2) + 3., 3));
					break;
				default:
					expected = 0.;
			}
			assertEquals(expected, actual, DELTA);
		}
	}
	/**
	 *
	 * @param v Degrees of Freedom
	 */
	private void testDLogPdfAtGivenDegreesOfFreedom(int v) {
		StudentTVertex studentT = new StudentTVertex(v, random);
		
		for(double t = -4.5; t <= 4.5; t += 0.5) {
			double expected;
			double actual = studentT.dLogPdf(t).get(studentT.getId()).scalar();
			switch(v) {
				case 1:
					expected = (-2 * t) / (pow(t, 2) + 1.);
					break;
				case 2:
					expected = (-3 * t) / (pow(t, 2) + 2.);
					break;
				case 3:
					expected = (-4 * t) / (pow(t, 2) + 3.);
					break;
				default:
					expected = 0.;
			}
			assertEquals(expected, actual, DELTA);
		}
	}
}
