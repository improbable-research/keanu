package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

public class DoubleReduceVertex extends NonProbabilisticDouble {
    protected final List<? extends Vertex<Double>> inputs;
    protected final BiFunction<Double, Double, Double> f;
    protected final Supplier<DualNumber> dualNumberSupplier;


    public DoubleReduceVertex(Collection<? extends Vertex<Double>> inputs, BiFunction<Double, Double, Double> f, Supplier<DualNumber> dualNumberSupplier) {
        this.inputs = new ArrayList<>(inputs);
        this.f = f;
        this.dualNumberSupplier = dualNumberSupplier;
        setParents(inputs);

        if (inputs.size() < 2) {
            throw new IllegalArgumentException("DoubleReduceVertex should have at least two input vertices, called with " + inputs.size());
        }
    }

    public DoubleReduceVertex(BiFunction<Double, Double, Double> f, Supplier<DualNumber> dualNumberSupplier, Vertex<Double>... input) {
        this(Arrays.asList(input), f, dualNumberSupplier);
    }

    public DoubleReduceVertex(List<? extends Vertex<Double>> inputs, BiFunction<Double, Double, Double> f) {
        this(inputs, f, null);
    }

    @Override
    public Double sample() {
        return applyReduce(Vertex::sample);
    }

    @Override
    public Double lazyEval() {
        setValue(applyReduce(Vertex::lazyEval));
        return getValue();
    }

    @Override
    public Double getDerivedValue() {
        return applyReduce(Vertex::getValue);
    }

    private double applyReduce(Function<Vertex<Double>, Double> mapper) {
        Iterator<? extends Vertex<Double>> samples = inputs.iterator();

        double c = samples.next().getValue();
        while (samples.hasNext()) {
            c = f.apply(c, mapper.apply(samples.next()));
        }

        return c;
    }

    @Override
    public DualNumber calcDualNumber(Map<Vertex, DualNumber> dualNumberMap) {
        if (dualNumberSupplier != null) {
            return dualNumberSupplier.get();
        }

        throw new UnsupportedOperationException();
    }
}
