package io.improbable.keanu.vertices.model;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;

import java.io.IOException;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * A vertex whose operation is the execution of a command line process.
 *
 * It is able to execute this process and parse the result.
 *
 * It stores multiple output values and a model result vertex is required to extract a specific value.
 */
public class ProcessModelVertex {

    public static LambdaModelVertex create(Map<VertexLabel, Vertex<? extends Tensor>> inputs,
                                           String command,
                                           BiFunction<Map<VertexLabel, Vertex<? extends Tensor>>, String, String> commandForExecution,
                                           Function<Map<VertexLabel, Vertex<? extends Tensor>>, Map<VertexLabel, Tensor>> updateValues) {
        return new LambdaModelVertex(inputs, i -> {
            String newCommand = commandForExecution.apply(inputs, command);
            try {
                Process cmd = Runtime.getRuntime().exec(newCommand);
                cmd.waitFor();
            } catch (IOException | InterruptedException e) {
                throw new RuntimeException("Failed during execution of the process. " + e);
            }
        }, updateValues);
    }
}
