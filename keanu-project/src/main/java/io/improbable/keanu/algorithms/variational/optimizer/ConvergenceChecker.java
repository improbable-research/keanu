package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface ConvergenceChecker {

    enum Norm {
        L2, MAX_ABS
    }

    boolean hasConverged(DoubleTensor[] position, DoubleTensor[] nextPosition);

    static double l2Norm(DoubleTensor[] a) {
        double magPow2 = 0;
        for (int i = 0; i < a.length; i++) {
            magPow2 += a[i].pow(2).sum();
        }

        return Math.sqrt(magPow2);
    }

    static double maxAbs(DoubleTensor[] a) {

        double max = -Double.MAX_VALUE;
        for (int i = 0; i < a.length; i++) {
            max = Math.max(max, a[i].max());
        }

        return max;
    }

    static ConvergenceChecker absoluteChecker(double threshold) {
        return absoluteChecker(Norm.MAX_ABS, threshold);
    }

    static ConvergenceChecker absoluteChecker(Norm norm, double threshold) {
        return new AbsoluteConvergenceChecker(norm, threshold);
    }

    static ConvergenceChecker relativeChecker(double threshold) {
        return relativeChecker(Norm.MAX_ABS, threshold);
    }

    static ConvergenceChecker relativeChecker(Norm norm, double threshold) {
        return new RelativeConvergenceChecker(norm, threshold);
    }
}
