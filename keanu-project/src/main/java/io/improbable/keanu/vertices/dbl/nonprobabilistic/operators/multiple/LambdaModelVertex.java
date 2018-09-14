package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
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
    private Map<VertexLabel, Tensor> outputs;
    private Consumer<Map<VertexLabel, Vertex<? extends Tensor>>> executor;
    private Function<Map<VertexLabel, Vertex<? extends Tensor>>, Map<VertexLabel, Tensor>> extractOutput;
    private boolean hasCalculated;

    public LambdaModelVertex(Map<VertexLabel, Vertex<? extends Tensor>> inputs,
                             Consumer<Map<VertexLabel, Vertex<? extends Tensor>>> executor,
                             Function<Map<VertexLabel, Vertex<? extends Tensor>>, Map<VertexLabel, Tensor>> extractOutput) {
        this.inputs = inputs;
        this.outputs = Collections.EMPTY_MAP;
        this.executor = executor;
        this.extractOutput = extractOutput;
        this.hasCalculated = false;
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
        hasCalculated = true;
    }

    @Override
    public Map<VertexLabel, Tensor> updateValues(Map<VertexLabel, Vertex<? extends Tensor>> inputs) {
        outputs = extractOutput.apply(inputs);
        return outputs;
    }

    @Override
    public boolean hasCalculated() {
        return false;
    }

    @Override
    public DoubleTensor getDoubleModelOutputValue(VertexLabel label) {
        return (DoubleTensor) outputs.get(label);
    }

    @Override
    public IntegerTensor getIntegerModelOutputValue(VertexLabel label) {
        return (IntegerTensor) outputs.get(label);
    }

    @Override
    public BooleanTensor getBooleanModelOutputValue(VertexLabel label) {
        return (BooleanTensor) outputs.get(label);
    }

}
