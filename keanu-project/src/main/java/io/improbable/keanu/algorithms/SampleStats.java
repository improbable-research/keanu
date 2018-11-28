package io.improbable.keanu.algorithms;

import java.util.Arrays;
import java.util.NoSuchElementException;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

public class SampleStats {
    public static double[] acf(double[] samples) {
        double[] result = new double[samples.length];
        double[] acovResult = acov(samples);
        for (int i = 0; i < samples.length; i++) {
            result[i] = acovResult[i] / acovResult[0];
        }
        return result;
    }

    public static double[] acov(double[] samples) {
        double mean = Arrays.stream(samples).average().orElseThrow(NoSuchElementException::new);
        double[] demean = Arrays.stream(samples).map(x -> x - mean).toArray();
        int n = demean.length;

        // Zero padding needed to stop mxing of convolution results
        // See last paragraph of https://dsp.stackexchange.com/a/745
        int fftSize = nextPowerOfTwo(2 * n + 1);
        double[] demeanPaddedWithZeros = Arrays.copyOf(demean, fftSize);

        FastFourierTransformer ffTransformer = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex fftData[] = ffTransformer.transform(demeanPaddedWithZeros, TransformType.FORWARD);
        Complex fftMultipliedWithConj[] = multiplyWithConjugate(fftData);
        Complex ifft[] = ffTransformer.transform(fftMultipliedWithConj, TransformType.INVERSE);
        Complex truncated[] = Arrays.copyOf(ifft, n);
        double realResult[] = getRealParts(truncated);

        realResult = Arrays.stream(realResult).map(x -> x/n).toArray();
        return realResult;
    }


    private static double[] padWithZeroes(double[] values, int newLength) {
        //TODO: Check padding size?
        return Arrays.copyOf(values, newLength);
    }

    private static int nextPowerOfTwo(int x) {
        int highestOneBit = Integer.highestOneBit(x);
        if (x == highestOneBit) {
            return x;
        }
        return highestOneBit << 1;

    }

    private static double[] getRealParts(Complex[] complexNumbers) {
        double[] reals = new double[complexNumbers.length];
        for (int i = 0; i < complexNumbers.length; i++) {
            reals[i] = complexNumbers[i].getReal();
        }
        return reals;
    }

    private static Complex[] multiplyWithConjugate(Complex[] complexNumbers) {
        Complex[] result = new Complex[complexNumbers.length];
        for (int i = 0; i < complexNumbers.length; i++) {
            result[i] = complexNumbers[i].multiply(complexNumbers[i].conjugate());
        }
        return result;
    }
}
