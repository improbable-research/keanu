package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Map;

public class Flip extends ProbabilisticBool {

    private final Vertex<Double> probTrue;

    public Flip(Vertex<Double> probTrue) {
        this.probTrue = probTrue;
        setValue(false);
        setParents(probTrue);
    }

    public Flip(double probTrue) {
        this(new ConstantDoubleVertex(probTrue));
    }

    public Vertex<Double> getProbTrue() {
        return probTrue;
    }

    @Override
    public double logPmf(Boolean value) {
        final double probability = value ? probTrue.getValue() : 1 - probTrue.getValue();
        return Math.log(probability);
    }

    @Override
    public Map<Long, DoubleTensor> dLogPmf(Boolean value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Boolean sample(KeanuRandom random) {
        return random.nextDouble() < probTrue.getValue();
    }

}
