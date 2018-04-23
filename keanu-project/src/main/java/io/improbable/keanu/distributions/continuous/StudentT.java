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
	public static double sample(int v, Random random) {
		assert( v > 0 );
		return Guassian.sample( 0., 1., random) / sqrt( ChiSquared.sample(v, random) / v );
		
	}
	
	public static double sample(int v, double mu, double sigma, Random random) {
		assert( v > 0 );
		return Guassian.sample( mu, sigma, random) / sqrt( ChiSquared.sample(v, random) / v );
		
	}
	
	public static double pdf(int v, Random random) {
		final double halfVPlusOne = ((double) v + 1.) / 2.;
		final double halfV = (double) v / 2.;
		final double numerator = gamma(halfVPlusOne);
		final double denominator = sqrt((double) v * PI) * gamma(halfV);
		final double multiplier = pow(1. + pow((double) random, 2), -halfVPlusOne);
		
		return (numerator / denominator) * multiplier;
	}
	
	public static double pdf(int v, double mu, double sigma, Random random) {
		final double halfVPlusOne = ((double) v + 1.) / 2.;
		final double halfV = (double) v / 2.;
		final double numerator = gamma(halfVPlusOne);
		final double denominator = sqrt((double) v * PI) * gamma(halfV);
		final double multiplier = pow(1. + pow((double) random, 2), -halfVPlusOne);
		
		return (numerator / denominator) * multiplier;
	}
}
