package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Map;
import java.util.Random;

public class Flip extends ProbabilisticBool {

    private final Vertex<Double> probabilityTrue;
    private final Random random;

    public Flip(Vertex<Double> probabilityTrue, Random random) {
        this.probabilityTrue = probabilityTrue;
        this.random = random;
        setValue(false);
        setParents(probabilityTrue);
    }

    public Flip(double probabilityTrue, Random random) {
        this(new ConstantDoubleVertex(probabilityTrue), random);
    }

    public Flip(double probabilityTrue) {
        this(new ConstantDoubleVertex(probabilityTrue), new Random());
    }

    public Vertex<Double> getProbabilityTrue() {
        return probabilityTrue;
    }

    @Override
    public double logProb(Boolean value) {
        final double probability = value ? probabilityTrue.getValue() : 1 - probabilityTrue.getValue();
        return Math.log(probability);
    }

    @Override
    public Map<String, Double> dLogProb(Boolean value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Boolean sample() {
        return random.nextDouble() < probabilityTrue.getValue();
    }

}
