package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

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

public class LambdaModelVertex extends DoubleVertex implements ModelVertex<DoubleTensor> {

    private Map<VertexLabel, Vertex<? extends Tensor>> inputs;
    private Map<VertexLabel, Object> outputs;
    private Consumer<Map<VertexLabel, Vertex<? extends Tensor>>> executor;
    private Function<Map<VertexLabel, Vertex<? extends Tensor>>, Map<VertexLabel, Object>> extractOutput;

    public LambdaModelVertex(Map<VertexLabel, Vertex<? extends Tensor>> inputs,
                             Consumer<Map<VertexLabel, Vertex<? extends Tensor>>> executor,
                             Function<Map<VertexLabel, Vertex<? extends Tensor>>, Map<VertexLabel, Object>> extractOutput) {
        this.inputs = inputs;
        this.outputs = Collections.EMPTY_MAP;
        this.executor = executor;
        this.extractOutput = extractOutput;
        setParents(inputs.entrySet().stream().map(r -> r.getValue()).collect(Collectors.toList()));
    }

    @Override
    public DoubleTensor calculate() {
        run(inputs);
        updateValues(inputs);
        return DoubleTensor.scalar(0.);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        for (Map.Entry<VertexLabel, Vertex<? extends Tensor>> input : inputs.entrySet()) {
            input.getValue().sample();
        }
        return calculate();
    }

    @Override
    public void run(Map<VertexLabel, Vertex<? extends Tensor>> inputs) {
        executor.accept(inputs);
    }

    public Map<VertexLabel, Object> setValue(Map<VertexLabel, Object> values) {
        outputs = values;
        return outputs;
    }

    @Override
    public Map<VertexLabel, Object> updateValues(Map<VertexLabel, Vertex<? extends Tensor>> inputs) {
        outputs = extractOutput.apply(inputs);
        return outputs;
    }

    @Override
    public Double getDoubleModelOutputValue(VertexLabel label) {
        return (Double) outputs.get(label);
    }

    @Override
    public Integer getIntegerModelOutputValue(VertexLabel label) {
        return (Integer) outputs.get(label);
    }

    @Override
    public Boolean getBooleanModelOutputValue(VertexLabel label) {
        return (Boolean) outputs.get(label);
    }

}
