package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

import java.util.Map;

public class UniformIntVertex extends ProbabilisticInteger {

    private Vertex<Integer> min;
    private Vertex<Integer> max;

    /**
     * @param min The inclusive lower bound.
     * @param max The exclusive upper bound.
     */
    public UniformIntVertex(Vertex<Integer> min, Vertex<Integer> max) {
        this.min = min;
        this.max = max;
        setParents(min, max);
    }

    public UniformIntVertex(int min, int max) {
        this(new ConstantIntegerVertex(min), new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(Vertex<Integer> min, int max) {
        this(min, new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(int min, Vertex<Integer> max) {
        this(new ConstantIntegerVertex(min), max);
    }

    public Vertex<Integer> getMin() {
        return min;
    }

    public Vertex<Integer> getMax() {
        return max;
    }

    @Override
    public double logPmf(Integer value) {
        final double probability = 1.0 / (max.getValue() - min.getValue());
        return Math.log(probability);
    }

    @Override
    public Map<Long, DoubleTensor> dLogPmf(Integer value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Integer sample(KeanuRandom random) {
        return min.getValue() + random.nextInt(max.getValue() - min.getValue());
    }
}
