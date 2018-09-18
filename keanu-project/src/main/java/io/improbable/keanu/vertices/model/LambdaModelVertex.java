package io.improbable.keanu.vertices.model;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Collections;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * A vertex whose operation is the execution of a lambda.
 *
 * It is able to execute a lambda and is able to parse the result.
 *
 * It stores multiple output values and a model result vertex is required to extract a specific value.
 */
public class LambdaModelVertex extends DoubleVertex implements ModelVertex<DoubleTensor> {

    private Map<VertexLabel, Vertex<? extends Tensor>> inputs;
    private Map<VertexLabel, Tensor> outputs;
    private Consumer<Map<VertexLabel, Vertex<? extends Tensor>>> executor;
    private Function<Map<VertexLabel, Vertex<? extends Tensor>>, Map<VertexLabel, Tensor>> extractOutput;
    private boolean hasValue;

    public LambdaModelVertex(Map<VertexLabel, Vertex<? extends Tensor>> inputs,
                             Consumer<Map<VertexLabel, Vertex<? extends Tensor>>> executor,
                             Function<Map<VertexLabel, Vertex<? extends Tensor>>, Map<VertexLabel, Tensor>> extractOutput) {
        this.inputs = inputs;
        this.outputs = Collections.EMPTY_MAP;
        this.executor = executor;
        this.extractOutput = extractOutput;
        this.hasValue = false;
        setParents(inputs.entrySet().stream().map(r -> r.getValue()).collect(Collectors.toList()));
    }

    @Override
    public DoubleTensor calculate() {
        run();
        updateValues(inputs);
        return DoubleTensor.scalar(0.0);
    }

    @Override
    public boolean hasValue() {
        return hasValue;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        for (Vertex<? extends Tensor> input : inputs.values()) {
            input.sample();
        }
        calculate();
        return DoubleTensor.scalar(0.0);
    }

    @Override
    public void run() {
        executor.accept(inputs);
        hasValue = true;
    }

    @Override
    public Map<VertexLabel, Tensor> updateValues(Map<VertexLabel, Vertex<? extends Tensor>> inputs) {
        outputs = extractOutput.apply(inputs);
        return outputs;
    }

    @Override
    public boolean hasCalculated() {
        return hasValue;
    }

    @Override
    public <U, T extends Tensor<U>> T getModelOutputValue(VertexLabel label) {
        return (T) outputs.get(label);
    }

}
