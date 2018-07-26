package io.improbable.keanu.algorithms.variational;

import java.util.List;

import static org.nd4j.linalg.util.MathUtils.sum;

public class Histogram {
    //this.bins = linspace(samples.min(), samples.max(), nBins);
    //this.hist = normalizeHistogram(buildHistogram(samples, bins));

    private static double[] normalizeHistogram(double[] hist){
        double sum = sum(hist);
        for (int i = 0; i < hist.length; i++){
            hist[i] /= sum;
        }
        return hist;
    }

    private static double[] listToArray(List<Double> l){
        double[] outArr = new double[l.size()];
        for (int i = 0; i < l.size(); i++){
            outArr[i] = l.get(i);
        }
        return outArr;
    }

    private static double[] buildHistogram(double[] samples, double[] bins){
        double[] hist = new double[bins.length];
        for (int i = 0; i < bins.length; i++){
            hist[i] = 0;
        }

        for (double s:samples) {
            for (int i = 1; i < bins.length; i++) {
                if ((bins[i] > s) && (bins[i - 1] <= s)) {
                    hist[i]++;
                    break;
                }
            }
        }
        return hist;
    }
}
