package io.improbable.keanu.benchmarks;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.keanu.KeanuComputableGraph;
import io.improbable.keanu.backend.keanu.compiled.KeanuCompiledGraphBuilder;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static java.util.concurrent.TimeUnit.MILLISECONDS;

@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1000, timeUnit = MILLISECONDS)
@Measurement(iterations = 5, time = 1000, timeUnit = MILLISECONDS)
@Fork(3)
public class CompiledBackEnds {

    public enum Backend {
        KEANU_GRAPH, KEANU_COMPILED
    }

    @Param({"KEANU_COMPILED", "KEANU_GRAPH"})
    public Backend backend;

    @Param({"100", "0"})
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
            case KEANU_GRAPH:
                computableGraph = keaunuGraph(outputNode);
                break;
            case KEANU_COMPILED:
                computableGraph = compiledKeaunuGraph(outputNode);
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
        compiler.registerOutput(output.getReference());

        return compiler.build();
    }

    public ComputableGraph keaunuGraph(DoubleVertex output) {

        List<IVertex> toposortedGraph = output.getConnectedGraph().stream()
            .sorted(Comparator.comparing(IVertex::getId, Comparator.naturalOrder()))
            .collect(Collectors.toList());

        return new KeanuComputableGraph(toposortedGraph, ImmutableSet.of(output));
    }

    @Benchmark
    public DoubleTensor sweepValues() {
        DoubleTensor result = null;

        for (int i = 0; i < 10000; i++) {
            result = (DoubleTensor) computableGraph.compute(inputs).get(output);
            incrementInputs();
        }

        return result;
    }

    private void incrementInputs() {
        valueA.plusInPlace(0.1);
        valueB.plusInPlace(0.1);
    }

}
