package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class DoubleReduceVertex extends NonProbabilisticDouble {
    private final List<? extends Vertex<DoubleTensor>> inputs;
    private final BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f;
    private final Supplier<DualNumber> dualNumberSupplier;

    public DoubleReduceVertex(int[] shape, Collection<? extends Vertex<DoubleTensor>> inputs, BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f, Supplier<DualNumber> dualNumberSupplier) {
        if (inputs.size() < 2) {
            throw new IllegalArgumentException("DoubleReduceVertex should have at least two input vertices, called with " + inputs.size());
        }

        this.inputs = new ArrayList<>(inputs);
        this.f = f;
        this.dualNumberSupplier = dualNumberSupplier;
        setParents(inputs);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public DoubleReduceVertex(int[] shape, BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f, Supplier<DualNumber> dualNumberSupplier, Vertex<DoubleTensor>... input) {
        this(shape, Arrays.asList(input), f, dualNumberSupplier);
    }

    public DoubleReduceVertex(int[] shape, List<? extends Vertex<DoubleTensor>> inputs, BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f) {
        this(shape, inputs, f, null);
    }

    /**
     * Reduce vertex that assumes shape shape as inputs
     *
     * @param f                  reduce function
     * @param dualNumberSupplier auto diff supplier
     * @param input              input vertices to reduce
     */
    public DoubleReduceVertex(BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f, Supplier<DualNumber> dualNumberSupplier, Vertex<DoubleTensor>... input) {
        this(TensorShapeValidation.checkAllShapesMatch(Arrays.stream(input).map(Vertex::getShape).collect(Collectors.toList())),
            Arrays.asList(input), f, dualNumberSupplier);
    }

    /**
     * Reduce vertex that assumes shape shape as inputs
     *
     * @param f      reduce function
     * @param inputs input vertices to reduce
     */
    public DoubleReduceVertex(List<? extends Vertex<DoubleTensor>> inputs, BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f) {
        this(TensorShapeValidation.checkAllShapesMatch(inputs.stream().map(Vertex::getShape).collect(Collectors.toList())),
            inputs, f, null
        );
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
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        if (dualNumberSupplier != null) {
            return dualNumberSupplier.get();
        }

        throw new UnsupportedOperationException();
    }
}
