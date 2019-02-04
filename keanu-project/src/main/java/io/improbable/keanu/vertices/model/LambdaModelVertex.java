package io.improbable.keanu.vertices.model;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Supplier;


public class LambdaModelVertex extends DoubleVertex implements ModelVertex<DoubleTensor>, NonSaveableVertex {

    private Map<VertexLabel, Vertex<? extends Tensor>> inputs;
    private Map<VertexLabel, Vertex<? extends Tensor>> outputs;
    private Consumer<Map<VertexLabel, Vertex<? extends Tensor>>> executor;
    private Supplier<Map<VertexLabel, Vertex<? extends Tensor>>> extractOutput;
    private boolean hasValue;

    /**
     * A vertex whose operation is the execution of a lambda.
     * It is able to execute a lambda and is able to parse the result.
     * It stores multiple output values in a map.
     * Use a ModelResultVertex to extract a value by label from this vertex.
     *
     * @param inputs       input vertices to the model
     * @param executor     the operation to perform
     * @param updateValues a function to extract the output values (once the operation has been performed) and update
     *                     the models output values.
     */
    public LambdaModelVertex(Map<VertexLabel, Vertex<? extends Tensor>> inputs,
                             Consumer<Map<VertexLabel, Vertex<? extends Tensor>>> executor,
                             Supplier<Map<VertexLabel, Vertex<? extends Tensor>>> updateValues) {
        super(Tensor.SCALAR_SHAPE);
        this.inputs = inputs;
        this.outputs = Collections.emptyMap();
        this.executor = executor;
        this.extractOutput = updateValues;
        this.hasValue = false;
        setParents(inputs.values());
    }

    /**
     * A vertex whose operation is the execution of a command line process.
     * It is able to execute this process and parse the result.
     * It stores multiple output values in a map.
     * Use a ModelResultVertex to extract a value by label from this vertex.
     *
     * @param inputs       input vertices to the model
     * @param command      the command to execute
     * @param updateValues a function to extract the output values (once the operation has been performed) and update
     *                     the models output values.
     * @return a process model vertex
     */
    @SuppressWarnings("squid:S2142")    // "InterruptedException" should not be ignored
    public static LambdaModelVertex createFromProcess(Map<VertexLabel, Vertex<? extends Tensor>> inputs,
                                                      String command,
                                                      Supplier<Map<VertexLabel, Vertex<? extends Tensor>>> updateValues) {
        return new LambdaModelVertex(inputs, i -> {
            try {
                Process cmd = Runtime.getRuntime().exec(command);
                cmd.waitFor();
            } catch (IOException | InterruptedException e) {
                throw new RuntimeException("Failed during execution of the process. " + e);
            }
        }, updateValues);
    }

    /**
     * This vertex stores multiple values in a key value pair of label to result.
     * As a result it should never be asked for its value directly.
     * Use a ModelResultVertex to extract a value from this vertex by label.
     *
     * @return a placeholder value
     */
    @Override
    public DoubleTensor calculate() {
        run();
        updateValues();
        return DoubleTensor.scalar(0.0);
    }

    @Override
    public boolean hasValue() {
        return hasValue;
    }

    @Override
    public void run() {
        executor.accept(inputs);
        hasValue = true;
    }

    @Override
    public Map<VertexLabel, Vertex<? extends Tensor>> updateValues() {
        outputs = extractOutput.get();
        return outputs;
    }

    @Override
    public boolean hasCalculated() {
        return hasValue();
    }

    @Override
    public <U, T extends Tensor<U>> T getModelOutputValue(VertexLabel label) {
        return (T) outputs.get(label).getValue();
    }

}
