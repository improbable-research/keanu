package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.op.Scope;

import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

public class TensorflowComputableGraphFactory {

    public static TensorflowComputableGraph convert(Set<Vertex> vertices) {
        return convert(vertices, new HashMap<>());
    }

    public static TensorflowComputableGraph convert(Collection<? extends Vertex> vertices, Map<Vertex<?>, Output<?>> lookup) {

        Graph graph = new Graph();
        Scope scope = new Scope(graph);
        TensorflowOpHelper graphBuilder = new TensorflowOpHelper(scope);

        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(Comparator.comparing(Vertex::getId, Comparator.naturalOrder()));
        priorityQueue.addAll(vertices);

        Map<VariableReference, Object> latentVariables = new HashMap<>();

        Vertex visiting;
        while ((visiting = priorityQueue.poll()) != null) {

            Output<?> visitingConverted;
            if (visiting instanceof Probabilistic) {
                if (visiting.isObserved()) {
                    visitingConverted = TensorflowGraphConverter.createConstant(visiting, graphBuilder);
                } else {
                    visitingConverted = TensorflowGraphConverter.createVariable(visiting, graphBuilder);
                    latentVariables.put(visiting.getReference(), visiting.getValue());
                }
            } else {
                TensorflowGraphConverter.OpMapper vertexMapper = TensorflowGraphConverter.opMappers.get(visiting.getClass());

                if (vertexMapper == null) {
                    throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " not supported for Tensorflow conversion");
                }

                visitingConverted = vertexMapper.apply(visiting, lookup, graphBuilder);
            }

            lookup.put(visiting, visitingConverted);
        }

        return new TensorflowComputableGraph(new Session(scope.graph()), scope, latentVariables);
    }
}
