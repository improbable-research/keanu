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
	public static double sample(int t, Random random) {
		assert( t >= 1 );
		return Guassian.sample( 0., 1., random) / sqrt( ChiSquared.sample(t, random) / t );
		
	}
	
	public static double pdf(int t, int v) {
		final double halfVPlusOne = ((double) v + 1.) / 2.;
		final double halfV = (double) v / 2.;
		final double numerator = gamma(halfVPlusOne);
		final double denominator = sqrt((double) v * PI) * gamma(halfV);
		final double multiplier = pow(1. + pow((double) t, 2), -halfVPlusOne);
		
		return (numerator / denominator) * multiplier;
	}
}
