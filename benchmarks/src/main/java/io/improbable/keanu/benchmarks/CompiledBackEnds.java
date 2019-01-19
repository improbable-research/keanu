package io.improbable.keanu.benchmarks;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.backend.keanu.KeanuCompiledGraphBuilder;
import io.improbable.keanu.backend.keanu.KeanuComputableGraph;
import io.improbable.keanu.backend.tensorflow.TensorflowComputableGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.openjdk.jmh.annotations.*;

import java.util.HashMap;
import java.util.Map;

@State(Scope.Benchmark)
public class CompiledBackEnds {

    public enum Backend {
        KEANU_GRAPH, KEANU_COMPILED, TENSORFLOW, PRECOMPILED_KEANU
    }

    @Param({"PRECOMPILED_KEANU", "KEANU_COMPILED", "TENSORFLOW", "KEANU_GRAPH"})
    public Backend backend;

    public ComputableGraph computableGraph;
    public Map<VariableReference, DoubleTensor> inputs;

    public DoubleTensor valueA;
    public DoubleTensor valueB;

    public VariableReference output;

    @Setup
    public void setup() {

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);

        valueA = DoubleTensor.zeros(100, 100);
        valueB = DoubleTensor.create(0.5, new long[]{100, 100});

        inputs = new HashMap<>();
        inputs.put(A.getReference(), valueA);
        inputs.put(B.getReference(), valueB);

        DoubleVertex outputNode = getGraphOutputNode(A, B);

        output = outputNode.getReference();

        switch (backend) {
            case PRECOMPILED_KEANU:
                computableGraph = precompiledKeaunuGraph(outputNode);
                break;
            case KEANU_GRAPH:
                computableGraph = keaunuGraph(outputNode);
                break;
            case KEANU_COMPILED:
                computableGraph = compiledKeaunuGraph(outputNode);
                break;
            case TENSORFLOW:
                computableGraph = tensorflowGraph(outputNode);
                break;
        }
    }

    public DoubleVertex getGraphOutputNode(DoubleVertex A, DoubleVertex B) {

        DoubleVertex C = A.times(B);

        return C;
    }

    public ComputableGraph compiledKeaunuGraph(DoubleVertex output) {
        KeanuCompiledGraphBuilder compiler = new KeanuCompiledGraphBuilder();

        compiler.convert(output.getConnectedGraph());
        compiler.registerOutput(output.getReference());

        return compiler.build();
    }

    public ComputableGraph keaunuGraph(DoubleVertex output) {
        return new KeanuComputableGraph(output.getConnectedGraph());
    }

    public ComputableGraph precompiledKeaunuGraph(DoubleVertex output) {
        return new PreCompiledKeaunGraph(output.getReference());
    }

    public ComputableGraph tensorflowGraph(DoubleVertex output) {
        return TensorflowComputableGraph.convert(output.getConnectedGraph());
    }

    @Benchmark
    public DoubleTensor sweepValues() {
        DoubleTensor result = null;

        for (int i = 0; i < 10000; i++) {
            result = computableGraph.compute(inputs, output);
            incrementInputs();
        }

        return result;
    }

    private void incrementInputs() {
        valueA.plusInPlace(0.1);
        valueB.plusInPlace(0.1);
    }

}
