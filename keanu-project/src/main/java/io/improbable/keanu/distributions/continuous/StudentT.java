package io.improbable.keanu.distributions.continuous;

import java.util.Random;

import static java.lang.Math.*;
import static org.apache.commons.math3.special.Gamma.gamma;

public class StudentT {
	/**
	 * Computer Generation of Statistical Distributions
	 * by Richard Saucier
	 * ARL-TR-2168 March 2000
	 * 5.1.23 page 36
	 */
	public static double sample(double v, Random random) {
		assert( v > 0. );
		return Guassian.sample( 0., 1., random) / sqrt( ChiSquared.sample(v, random) / v );
		
	}
	
	public static double sample(double v, double mu, double sigma, Random random) {
		assert( v > 0. );
		return Guassian.sample( mu, sigma, random) / sqrt( ChiSquared.sample(v, random) / v );
		
	}
	
	public static double pdf(double v, double x) {
		final double halfVPlusOne = (v + 1.) / 2.;
		final double halfV = v / 2.;
		final double numerator = gamma(halfVPlusOne);
		final double denominator = sqrt(v * PI) * gamma(halfV);
		final double multiplier = pow(1. + pow((x, 2), -halfVPlusOne);
		
		return (numerator / denominator) * multiplier;
	}
	
	public static double pdf(double v, double mu, double sigma, double x) {
		return 0.;
	}
}
