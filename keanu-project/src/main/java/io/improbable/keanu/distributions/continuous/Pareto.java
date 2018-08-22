package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.Diffs.L;
import static io.improbable.keanu.distributions.dual.Diffs.S;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Pareto implements ContinuousDistribution {

    private final DoubleTensor location;
    private final DoubleTensor scale;

    public static ContinuousDistribution withParameters(DoubleTensor location, DoubleTensor scale) {

        return new Pareto(location, scale);
    }

    private Pareto(DoubleTensor location, DoubleTensor scale) {
        this.location = location;
        this.scale = scale;
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        DoubleTensor dLogPdx = scale.plus(1.0).divInPlace(x).unaryMinusInPlace();
        DoubleTensor dLogPdLoc = DoubleTensor.zeros(x.getShape()).plusInPlace(scale).divInPlace(location);
        DoubleTensor dLogPdScale = scale.reciprocal().plusInPlace(location.log()).minusInPlace(x.log());

        return new Diffs()
            .put(X, dLogPdx)
            .put(L, dLogPdLoc)
            .put(S, dLogPdScale);
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor result = DoubleTensor.create(1., shape);
        result = result.minusInPlace(random.nextDouble(shape)).powInPlace(scale.reciprocal()).reciprocal()
            .timesInPlace(location);

        return result;
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        DoubleTensor result;

        /*
         * If we've been passed invalid values, then return Negative Infinity for all values, else just return the
         * calculated value
         */
        if (!location.greaterThan(0.0).allTrue() || !scale.greaterThan(0.0).allTrue()) {
            result = DoubleTensor.create(Double.NEGATIVE_INFINITY, x.getShape());
        } else {
            DoubleTensor invalids = x.getLessThanOrEqualToMask(location);
            result = scale.log().plusInPlace(location.log().timesInPlace(scale))
                .minusInPlace(scale.plus(1.0).timesInPlace(x.log()));

            result.setWithMaskInPlace(invalids, Double.NEGATIVE_INFINITY);
        }

        return result;
    }
}
