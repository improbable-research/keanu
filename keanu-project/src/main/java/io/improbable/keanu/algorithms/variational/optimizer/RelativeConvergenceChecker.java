package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class RelativeConvergenceChecker implements ConvergenceChecker {

    private final Norm normStrategy;
    private final double threshold;
    private final double epsilon;

    public RelativeConvergenceChecker(final Norm normStrategy, double threshold) {
        this(normStrategy, threshold, 1e-6);
    }

    @Override
    public final boolean hasConverged(DoubleTensor[] position, DoubleTensor[] nextPosition) {

        DoubleTensor[] delta = relativeDelta(position, nextPosition, epsilon);

        return normStrategy.calculate(delta) < threshold;
    }

    private DoubleTensor[] relativeDelta(DoubleTensor[] a, DoubleTensor[] b, double epsilon) {

        DoubleTensor[] relative = new DoubleTensor[a.length];
        for (int i = 0; i < a.length; i++) {
            relative[i] = a[i].minus(b[i]).abs().plus(epsilon).div(a[i].plus(epsilon));
        }

        return relative;
    }

}
