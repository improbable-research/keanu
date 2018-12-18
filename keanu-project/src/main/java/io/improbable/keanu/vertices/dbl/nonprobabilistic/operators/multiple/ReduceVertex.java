package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

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

import static io.improbable.keanu.tensor.TensorShapeValidation.checkAllShapesMatch;

public class ReduceVertex extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor>, NonSaveableVertex {

    private final List<? extends Vertex<DoubleTensor>> inputs;
    private final BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> reduceFunction;
    private final Supplier<PartialDerivative> forwardModeAutoDiffLambda;
    private final Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda;

    public ReduceVertex(long[] shape, Collection<? extends Vertex<DoubleTensor>> inputs,
                        BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> reduceFunction,
                        Supplier<PartialDerivative> forwardModeAutoDiffLambda,
                        Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        super(shape);
        if (inputs.size() < 2) {
            throw new IllegalArgumentException("ReduceVertex should have at least two input vertices, called with " + inputs.size());
        }

        this.inputs = new ArrayList<>(inputs);
        this.reduceFunction = reduceFunction;
        this.forwardModeAutoDiffLambda = forwardModeAutoDiffLambda;
        this.reverseModeAutoDiffLambda = reverseModeAutoDiffLambda;
        setParents(inputs);
    }

    public ReduceVertex(long[] shape, BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> reduceFunction,
                        Supplier<PartialDerivative> forwardModeAutoDiffLambda,
                        Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda,
                        Vertex<DoubleTensor>... input) {
        this(shape, Arrays.asList(input), reduceFunction, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    public ReduceVertex(long[] shape, Collection<? extends Vertex<DoubleTensor>> inputs,
                        BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> reduceFunction) {
        this(shape, inputs, reduceFunction, null, null);
    }

    /**
     * Reduce vertex that assumes shape as inputs
     *
     * @param reduceFunction            reduce function
     * @param forwardModeAutoDiffLambda auto diff supplier
     * @param reverseModeAutoDiffLambda function for returning diff
     * @param input                     input vertices to reduce
     */
    public ReduceVertex(BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> reduceFunction,
                        Supplier<PartialDerivative> forwardModeAutoDiffLambda,
                        Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda,
                        Vertex<DoubleTensor>... input) {
        this(checkAllShapesMatch(Arrays.stream(input).map(Vertex::getShape).collect(Collectors.toList())),
            Arrays.asList(input), reduceFunction, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    /**
     * Reduce vertex that assumes shape as inputs
     *
     * @param reduceFunction reduce function
     * @param inputs         input vertices to reduce
     */
    public ReduceVertex(List<? extends Vertex<DoubleTensor>> inputs,
                        BiFunction<DoubleTensor, DoubleTensor, DoubleTensor> reduceFunction) {
        this(checkAllShapesMatch(inputs.stream().map(Vertex::getShape).collect(Collectors.toList())),
            inputs, reduceFunction, null, null
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
            result = reduceFunction.apply(result, mapper.apply(inputIterator.next()));
        }

        return result;
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInputs) {
        if (forwardModeAutoDiffLambda != null) {
            return forwardModeAutoDiffLambda.get();
        }

        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        return reverseModeAutoDiffLambda.apply(derivativeOfOutputWithRespectToSelf);
    }
}
