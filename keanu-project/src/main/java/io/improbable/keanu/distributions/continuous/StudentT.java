package io.improbable.keanu.distributions.continuous;

import java.util.Random;

import static java.lang.Math.*;
import static org.apache.commons.math3.special.Gamma.gamma;
import static org.apache.commons.math3.special.Gamma.digamma;

/**
 * Student T Distribution
 * https://en.wikipedia.org/wiki/Student%27s_t-distribution#Sampling_distribution
 *
 */
public class StudentT {
	/**
	 * Computer Generation of Statistical Distributions
	 * by Richard Saucier
	 * ARL-TR-2168 March 2000
	 * 5.1.23 page 36
	 *
	 * @param v Degrees of Freedom
	 * @param random random number generator (RNG) seed
	 * @return sample of Student T distribution
	 */
	public static double sample(int v, Random random) {
		if(v > 0) {
			throw new IllegalArgumentException("Invalid degrees of freedom (v), expect v > 0");
		}
		return Gaussian.sample( 0., 1., random) / sqrt( ChiSquared.sample((int) v, random) / v );
		
	}
	/**
	 *
	 * @param v Degrees of Freedom
	 * @param t random variable
	 * @return the Probability Density Function
	 */
	public static double pdf(int v, double t) {
		double halfVPlusOne = (v + 1.) / 2.;
		double halfV = v / 2.;
		double numerator = gamma(halfVPlusOne);
		double denominator = sqrt(v * PI) * gamma(halfV);
		double multiplier = pow(1. + (pow(t, 2) / v), -halfVPlusOne);
		
		return (numerator / denominator) * multiplier;
	}
	/**
	 *
	 * @param v Degrees of Freedom
	 * @param t random variable
	 * @return Log of the Probability Density Function
	 */
	public static double logPdf(int v, double t) {
		return log(pdf(v, t));
	}
	/**
	 *
	 * @param in input
	 * @return zeroth derivative of the digamma
	 */
	private static double zeroDerivativeOfDigamma(double in) {
		return digamma(in);
	}
	/**
	 *
	 * @param v Degrees of Freedom
	 * @param t random variable
	 * @return Differential of the Probability Density Function
	 */
	public static Diff dPdf(int v, double t) {
		double gammaHalfV = gamma(v / 2.);
		double gammaHalfVPlusOne = gamma((v + 1.) / 2.);
		double sqrtVPi = sqrt(v * PI);
		double tSq = pow(t, 2);
		double tSqDividedByVPlusOne = (tSq / v) + 1.;
		double sqrtVPiGammaHalfV = sqrtVPi * gammaHalfV;
//		double sqrtPiGammaHalfV = sqrt(PI) * gammaHalfV;
//
//		double zeroDerivativeOfDigammaHalfV = zeroDerivativeOfDigamma(v / 2.);
//		double zeroDerivativeOfDigammaHalfVPlusOne = zeroDerivativeOfDigamma((v + 1.) / 2.);
//
//		double dPdv_multiplier = pow(tSqDividedByVPlusOne, (-v - 1.) / 2.) * gammaHalfVPlusOne;
//		double dPdv_pt1 = -1. / (2. * sqrtPiGammaHalfV * pow(v, 3. / 2.));
//		double dPdv_pt2_numerator = -(tSq * (-v - 1.) / (2. * pow(v, 2.) * tSqDividedByVPlusOne)) -
//				(0.5 * log(tSqDividedByVPlusOne));
//		double dPdv_pt2 = dPdv_pt2_numerator / sqrtVPiGammaHalfV;
//		double dPdv_pt3 = zeroDerivativeOfDigammaHalfV / (2. * sqrtVPiGammaHalfV);
//		double dPdv_pt4 = zeroDerivativeOfDigammaHalfVPlusOne / (2. * sqrtVPiGammaHalfV);
//		double dPdv = dPdv_multiplier * (dPdv_pt1 + dPdv_pt2 - dPdv_pt3 + dPdv_pt4);
		
		double dPdt_numerator = (t * (-v - 1)) * pow(tSqDividedByVPlusOne, -v / 2.) * gammaHalfVPlusOne;
		double dPdt_denominator = (tSq + v) * sqrt((tSq + v) / v) * sqrtVPiGammaHalfV;
		double dPdt = dPdt_numerator / dPdt_denominator;
		
		return new Diff(dPdt);
	}
	/**
	 *
	 * @param v Degrees of Freedom
	 * @param t random variable
	 * @return Differential of the Log of the Probability Density Function
	 */
	public static Diff dLogPdf(int v, double t) {
		double tSq = pow(t, 2);
		double tSqPlusV = tSq + v;
		
//		double digammaHalfV = zeroDerivativeOfDigamma(v / 2.);
//		double digammaVPlusHalf = zeroDerivativeOfDigamma((v + 1.) / 2.);
//
//		double digamma_pt = log(tSqPlusV / v) + digammaHalfV - digammaVPlusHalf;
//		double pt1 = -tSq * (digamma_pt - 1.);
//		double pt2 = v * (digamma_pt);
//
//		double numerator = pt1 + pt2 + 1;
//		double denominator = 2. * tSqPlusV;
//
//		double dPdv = numerator / denominator;
		double dPdt = (-t * (v + 1.)) / (pow(t, 2.) + v);
		
		return new Diff(dPdt);
	}
	/**
	 * Differential Equation Class to store result of d/dv and d/dt
	 */
	public static class Diff {
//		public double dPdv;
		public double dPdt;
		
		public Diff(double dPdt) {
//			this.dPdv = dPdv;
			this.dPdt = dPdt;
		}
	}
}
