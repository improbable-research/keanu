package io.improbable.keanu.distributions.continuous;

import static org.apache.commons.math3.special.Gamma.gamma;

public class StudentT {
	public static double pdf(double t, double v) {
		final double halfVPlusOne = (v + 1) / 2;
		final double halfV = v / 2;
		final double numerator = gamma(halfVPlusOne);
		final double denominator = Math.sqrt(v * Math.PI) * gamma(halfV);
		final double multiplier = Math.pow(1 + Math.pow(t, 2), -halfVPlusOne);
		
		return (numerator / denominator) * multiplier;
	}
}
