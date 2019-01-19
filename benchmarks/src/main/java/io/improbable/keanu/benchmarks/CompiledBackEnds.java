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

    @Param({"KEANU_COMPILED", "KEANU_GRAPH", "TENSORFLOW"})
    public Backend backend;

    @Param({"0", "100"})
    public int linkCount;

    public ComputableGraph computableGraph;
    public Map<VariableReference, DoubleTensor> inputs;

    public DoubleTensor valueA;
    public DoubleTensor valueB;

    public VariableReference output;

    @Setup
    public void setup() {

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);

        valueA = DoubleTensor.scalar(0);
        valueB = DoubleTensor.scalar(0.5);

        inputs = new HashMap<>();
        inputs.put(A.getReference(), valueA);
        inputs.put(B.getReference(), valueB);

        DoubleVertex outputNode = getGraphOutputNode(A, B, linkCount);

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

    public DoubleVertex getGraphOutputNode(DoubleVertex A, DoubleVertex B, int links) {

        DoubleVertex out = A.times(B);
        DoubleVertex left = A;
        DoubleVertex right = B;

        for (int i = 0; i < links; i++) {
            left = out.plus(left);
            right = out.minus(right);
            out = left.times(right);
        }

        return out;
    }

    public ComputableGraph compiledKeaunuGraph(DoubleVertex output) {
        KeanuCompiledGraphBuilder compiler = new KeanuCompiledGraphBuilder();

        compiler.convert(output.getConnectedGraph());
        compiler.registerOutput(output);

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
