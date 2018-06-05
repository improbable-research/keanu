package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.NonProbabilisticDoubleTensor;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

public class TensorDoubleReduceVertex extends NonProbabilisticDoubleTensor {
    private final List<? extends Vertex<DoubleTensor>> inputs;
    private final BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f;
    private final Supplier<TensorDualNumber> dualNumberSupplier;


    public TensorDoubleReduceVertex(Collection<? extends Vertex<DoubleTensor>> inputs, BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f, Supplier<TensorDualNumber> dualNumberSupplier) {
        this.inputs = new ArrayList<>(inputs);
        this.f = f;
        this.dualNumberSupplier = dualNumberSupplier;
        setParents(inputs);

        if (inputs.size() < 2) {
            throw new IllegalArgumentException("DoubleReduceVertex should have at least two input vertices, called with " + inputs.size());
        }
    }

    public TensorDoubleReduceVertex(BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f, Supplier<TensorDualNumber> dualNumberSupplier, Vertex<DoubleTensor>... input) {
        this(Arrays.asList(input), f, dualNumberSupplier);
    }

    public TensorDoubleReduceVertex(List<? extends Vertex<DoubleTensor>> inputs, BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f) {
        this(inputs, f, null);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return applyReduce(vertex -> vertex.sample(random));
    }

    @Override
    public DoubleTensor getDerivedValue() {
        return applyReduce(Vertex::getValue);
    }

    private DoubleTensor applyReduce(Function<Vertex<DoubleTensor>, DoubleTensor> mapper) {
        Iterator<? extends Vertex<DoubleTensor>> inputIterator = inputs.iterator();

        DoubleTensor result = inputIterator.next().getValue();
        while (inputIterator.hasNext()) {
            result = f.apply(result, mapper.apply(inputIterator.next()));
        }

        return result;
    }

    @Override
    public TensorDualNumber calculateDualNumber(Map<Vertex, TensorDualNumber> dualNumbers) {
        if (dualNumberSupplier != null) {
            return dualNumberSupplier.get();
        }

        throw new UnsupportedOperationException();
    }
}
