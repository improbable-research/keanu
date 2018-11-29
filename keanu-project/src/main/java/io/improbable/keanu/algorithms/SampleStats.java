package io.improbable.keanu.algorithms;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

import java.util.Arrays;
import java.util.NoSuchElementException;

public class SampleStats {
    public static double[] acf(double[] samples) {
        double[] acovResult = acov(samples);
        double variance = acovResult[0];
        double[] autocorr = Arrays.stream(acovResult).map(x -> x / variance).toArray();
        return autocorr;
    }

    public static double[] acov(double[] samples) {
        double mean = Arrays.stream(samples).average().orElseThrow(NoSuchElementException::new);
        double[] demean = Arrays.stream(samples).map(x -> x - mean).toArray();
        int n = demean.length;

        // Zero padding needed to stop mixing of convolution results
        // See last paragraph of https://dsp.stackexchange.com/a/745
        // FFT requires length to be power of two
        int fftSize = nextPowerOfTwo(2 * n + 1);
        double[] demeanPaddedWithZeros = Arrays.copyOf(demean, fftSize);

        Complex ifft[] = fftCrossCorrelationWitSelf(demeanPaddedWithZeros);
        double realParts[] = getRealPartsTruncated(ifft, n);
        double realPartsDivN[] = Arrays.stream(realParts).map(x -> x / n).toArray();

        return realPartsDivN;
    }

    private static Complex[] fftCrossCorrelationWitSelf(double[] values) {
        FastFourierTransformer ffTransformer = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex fftData[] = ffTransformer.transform(values, TransformType.FORWARD);
        Complex fftMultipliedWithConj[] = multiplyWithConjugateInPlace(fftData);
        Complex ifft[] = ffTransformer.transform(fftMultipliedWithConj, TransformType.INVERSE);
        return ifft;
    }


    private static int nextPowerOfTwo(int x) {
        int highestOneBit = Integer.highestOneBit(x);
        if (x == highestOneBit) {
            return x;
        }
        return highestOneBit << 1;

    }

    private static double[] getRealPartsTruncated(Complex[] complexNumbers, int newLength) {
        double[] reals = new double[newLength];
        for (int i = 0; i < newLength; i++) {
            reals[i] = complexNumbers[i].getReal();
        }
        return reals;
    }

    private static Complex[] multiplyWithConjugateInPlace(Complex[] complexNumbers) {
        for (int i = 0; i < complexNumbers.length; i++) {
            complexNumbers[i] = complexNumbers[i].multiply(complexNumbers[i].conjugate());
        }
        return complexNumbers;
    }
}
