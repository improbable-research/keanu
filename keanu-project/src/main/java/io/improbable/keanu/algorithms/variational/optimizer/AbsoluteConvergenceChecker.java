package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class AbsoluteConvergenceChecker implements ConvergenceChecker {

    private final Norm normStrategy;
    private final double threshold;

    @Override
    public final boolean hasConverged(DoubleTensor[] position, DoubleTensor[] nextPosition) {

        DoubleTensor[] delta = absoluteDelta(position, nextPosition);

        return normStrategy.calculate(delta) < threshold;
    }

    private DoubleTensor[] absoluteDelta(DoubleTensor[] a, DoubleTensor[] b) {

        DoubleTensor[] absolute = new DoubleTensor[a.length];
        for (int i = 0; i < a.length; i++) {
            absolute[i] = a[i].minus(b[i]).abs();
        }

        return absolute;
    }

}
