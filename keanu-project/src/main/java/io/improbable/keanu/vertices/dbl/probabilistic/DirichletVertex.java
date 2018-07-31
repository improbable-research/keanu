package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Dirichlet;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Map;

public class DirichletVertex extends ProbabilisticDouble {

    private final DoubleVertex concentration;

    public DirichletVertex(int[] tensorShape, DoubleVertex concentration) {
        //check shape

        this.concentration = concentration;
        setParents(concentration);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    @Override
    public double logPdf(DoubleTensor value) {
        return 0;
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        return null;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Dirichlet.withParameters(concentration.getValue()).sample(getShape(), random);
    }
}
