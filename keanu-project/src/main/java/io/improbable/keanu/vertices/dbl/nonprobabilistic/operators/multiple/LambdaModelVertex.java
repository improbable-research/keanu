package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleModelResultVertex;

import java.util.Collections;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

public class LambdaModelVertex extends DoubleVertex implements ModelVertex<DoubleTensor> {

    private Map<VertexLabel, DoubleVertex> inputs;
    private Map<VertexLabel, Double> outputs;
    private Consumer<Map<VertexLabel, DoubleVertex>> executor;
    private Function<Map<VertexLabel, DoubleVertex>, Map<VertexLabel, Double>> extractOutput;

    public LambdaModelVertex(Map<VertexLabel, DoubleVertex> inputs,
                             Consumer<Map<VertexLabel, DoubleVertex>> executor,
                             Function<Map<VertexLabel, DoubleVertex>, Map<VertexLabel, Double>> extractOutput) {
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
        for (Map.Entry<VertexLabel, DoubleVertex> input : inputs.entrySet()) {
            input.getValue().sample();
        }
        return calculate();
    }

    @Override
    public void run(Map<VertexLabel, DoubleVertex> inputs) {
        executor.accept(inputs);
    }

    @Override
    public Map<VertexLabel, Double> updateValues(Map<VertexLabel, DoubleVertex> inputs) {
        outputs = extractOutput.apply(inputs);
        return outputs;
    }

    @Override
    public Double getModelOutputValue(VertexLabel label) {
        return outputs.get(label);
    }

    @Override
    public DoubleVertex getModelOutputVertex(VertexLabel label) {
        return new DoubleModelResultVertex(this, label);
    }

}
