package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Dirichlet implements ContinuousDistribution {

    private final DoubleTensor concentration;
    private final int categories;

    public static ContinuousDistribution withParameters(DoubleTensor concentration) {
        return new Dirichlet(concentration, (int) concentration.getLength());
    }

    private Dirichlet(DoubleTensor concentration, int categories) {
        this.concentration = concentration;
        this.categories = categories;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        ContinuousDistribution gamma = Gamma.withParameters(concentration, DoubleTensor.ones(concentration.getShape()), DoubleTensor.zeros(concentration.getShape()));
        DoubleTensor gammaSamples = gamma.sample(concentration.getShape(), random);
        return normalise(gammaSamples);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        return null;
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        return null;
    }

    private DoubleTensor normalise(DoubleTensor gammaSamples) {
        double sum = gammaSamples.sum();
        return gammaSamples.div(sum);
    }
}
