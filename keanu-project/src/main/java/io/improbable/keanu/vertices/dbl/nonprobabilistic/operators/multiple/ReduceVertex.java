package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class ReduceVertex extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor> {
    private final List<? extends Vertex<DoubleTensor>> inputs;
    private final BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f;
    private final Supplier<DualNumber> dualNumberSupplier;

    public ReduceVertex(int[] shape, Collection<? extends Vertex<DoubleTensor>> inputs, BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f, Supplier<DualNumber> dualNumberSupplier) {
        if (inputs.size() < 2) {
            throw new IllegalArgumentException("ReduceVertex should have at least two input vertices, called with " + inputs.size());
        }

        this.inputs = new ArrayList<>(inputs);
        this.f = f;
        this.dualNumberSupplier = dualNumberSupplier;
        setParents(inputs);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public ReduceVertex(int[] shape, BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f, Supplier<DualNumber> dualNumberSupplier, Vertex<DoubleTensor>... input) {
        this(shape, Arrays.asList(input), f, dualNumberSupplier);
    }

    public ReduceVertex(int[] shape, List<? extends Vertex<DoubleTensor>> inputs, BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f) {
        this(shape, inputs, f, null);
    }

    /**
     * Reduce vertex that assumes shape as inputs
     *
     * @param f                  reduce function
     * @param dualNumberSupplier auto diff supplier
     * @param input              input vertices to reduce
     */
    public ReduceVertex(BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f, Supplier<DualNumber> dualNumberSupplier, Vertex<DoubleTensor>... input) {
        this(TensorShapeValidation.checkAllShapesMatch(Arrays.stream(input).map(Vertex::getShape).collect(Collectors.toList())),
            Arrays.asList(input), f, dualNumberSupplier);
    }

    /**
     * Reduce vertex that assumes shape as inputs
     *
     * @param f      reduce function
     * @param inputs input vertices to reduce
     */
    public ReduceVertex(List<? extends Vertex<DoubleTensor>> inputs, BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> f) {
        this(TensorShapeValidation.checkAllShapesMatch(inputs.stream().map(Vertex::getShape).collect(Collectors.toList())),
            inputs, f, null
        );
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return applyReduce(vertex -> vertex.sample(random));
    }

    @Override
    public DoubleTensor calculate() {
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
