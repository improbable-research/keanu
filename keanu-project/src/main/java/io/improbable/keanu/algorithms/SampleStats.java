package io.improbable.keanu.algorithms;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

import java.util.Arrays;
import java.util.NoSuchElementException;

public final class SampleStats {
    private static final FastFourierTransformer ffTransformer = new FastFourierTransformer(DftNormalization.STANDARD);

    public static double[] autocorrelation(double[] samples) {
        double[] acovResult = autocovariance(samples);
        double variance = acovResult[0];
        double[] autocorr = Arrays.stream(acovResult).map(x -> x / variance).toArray();
        return autocorr;
    }

    private static double[] autocovariance(double[] samples) {
        final int length = samples.length;
        double[] demeanPaddedWithZeros = calculatePaddedDemean(samples);
        Complex[] ifft = fftCrossCorrelationWithSelf(demeanPaddedWithZeros);
        double[] realParts = getRealPartsTruncated(ifft, length);
        double[] realPartsDivN = Arrays.stream(realParts).map(x -> x / length).toArray();
        return realPartsDivN;
    }

    private static double[] calculatePaddedDemean(double[] samples) {
        double[] demean = demean(samples);

        // Zero padding needed to stop mixing of convolution results
        // See last paragraph of https://dsp.stackexchange.com/a/745
        // FFT requires length to be power of two
        int fftSize = nextPowerOfTwo(2 * samples.length + 1);
        return Arrays.copyOf(demean, fftSize);
    }

    private static double[] demean(double[] samples) {
        double mean = Arrays.stream(samples).average().orElseThrow(NoSuchElementException::new);
        return Arrays.stream(samples).map(x -> x - mean).toArray();
    }

    private static int nextPowerOfTwo(int x) {
        int highestOneBit = Integer.highestOneBit(x);
        return (x == highestOneBit) ? x : highestOneBit << 1;
    }

    private static Complex[] fftCrossCorrelationWithSelf(double[] values) {
        Complex[] fftData = ffTransformer.transform(values, TransformType.FORWARD);
        Complex fftMultipliedWithConj[] = multiplyWithConjugateInPlace(fftData);
        Complex[] ifft = ffTransformer.transform(fftMultipliedWithConj, TransformType.INVERSE);
        return ifft;
    }

    private static Complex[] multiplyWithConjugateInPlace(Complex[] complexNumbers) {
        for (int i = 0; i < complexNumbers.length; i++) {
            complexNumbers[i] = complexNumbers[i].multiply(complexNumbers[i].conjugate());
        }
        return complexNumbers;
    }

    private static double[] getRealPartsTruncated(Complex[] complexNumbers, int newLength) {
        double[] reals = new double[newLength];
        for (int i = 0; i < newLength; i++) {
            reals[i] = complexNumbers[i].getReal();
        }
        return reals;
    }
}
