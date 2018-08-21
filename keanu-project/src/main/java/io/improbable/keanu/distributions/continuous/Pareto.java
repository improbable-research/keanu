package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.Diffs.L;
import static io.improbable.keanu.distributions.dual.Diffs.S;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Pareto implements ContinuousDistribution {

    private final DoubleTensor xm;
    private final DoubleTensor alpha;

    public static ContinuousDistribution withParameters(DoubleTensor xm, DoubleTensor alpha) {

        return new Pareto(xm, alpha);
    }

    private Pareto(DoubleTensor xm, DoubleTensor alpha) {
        this.xm = xm;
        this.alpha = alpha;
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        DoubleTensor dLogPdx = alpha.plus(1.0).divInPlace(x).timesInPlace(-1.0);
        DoubleTensor dLogPdxm = DoubleTensor.zeros(x.getShape()).plusInPlace(alpha).divInPlace(xm);
        DoubleTensor dLogPdalpha = alpha.reciprocal().plusInPlace(xm.log()).minusInPlace(x.log());

        return new Diffs()
            .put(X, dLogPdx)
            .put(L, dLogPdxm)
            .put(S, dLogPdalpha);
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor result = DoubleTensor.create(1., shape);
        result = result.minusInPlace(random.nextDouble(shape)).powInPlace(alpha.reciprocal()).reciprocal()
            .timesInPlace(xm);

        return result;
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        DoubleTensor result;

        /*
         * If we've been passed invalid values, then return Negative Infinity for all values, else just return the
         * calculated value
         */
        if (!xm.greaterThan(0).allTrue() || !alpha.greaterThan(0).allTrue()) {
            result = DoubleTensor.create(Double.NEGATIVE_INFINITY, x.getShape());
        } else {
            DoubleTensor invalids = x.getLessThanOrEqualToMask(xm);
            result = alpha.log().plusInPlace(xm.log().timesInPlace(alpha))
                .minusInPlace(alpha.plus(1.0).timesInPlace(x.log()));

            result.setWithMaskInPlace(invalids, Double.NEGATIVE_INFINITY);
        }

        return result;
    }
}
