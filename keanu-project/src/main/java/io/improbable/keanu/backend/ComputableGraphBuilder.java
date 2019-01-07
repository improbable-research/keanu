package io.improbable.keanu.backend;

import io.improbable.keanu.vertices.PlaceHolderVertex;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.Comparator;
import java.util.Map;
import java.util.PriorityQueue;

public interface ComputableGraphBuilder<T extends ComputableGraph> {

    void createConstant(Vertex visiting);

    void createVariable(Vertex visiting);

    void create(Vertex visiting);

    void connect(Map<Vertex<?>, Vertex<?>> connections);

    Collection<VariableReference> getLatentVariables();

    VariableReference add(VariableReference left, VariableReference right);

    T build();

    default void convert(Collection<? extends Vertex> vertices) {

        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(
            Comparator.comparing(Vertex::getId, Comparator.naturalOrder())
        );

        priorityQueue.addAll(vertices);

        Vertex visiting;
        while ((visiting = priorityQueue.poll()) != null) {

            if (visiting instanceof PlaceHolderVertex) {
                continue;
            }

            if (visiting instanceof Probabilistic) {
                if (visiting.isObserved()) {
                    createConstant(visiting);
                } else {
                    createVariable(visiting);
                }
            } else {
                create(visiting);
            }

        }
    }

}
