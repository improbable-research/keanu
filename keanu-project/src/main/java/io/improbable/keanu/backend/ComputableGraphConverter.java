package io.improbable.keanu.backend;

import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.Comparator;
import java.util.PriorityQueue;

public class ComputableGraphConverter {

    public static <T extends ComputableGraph> T convert(Collection<? extends Vertex> vertices, GraphBuilder<T> graphBuilder) {

        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(
            Comparator.comparing(Vertex::getId, Comparator.naturalOrder())
        );

        priorityQueue.addAll(vertices);

        Vertex visiting;
        while ((visiting = priorityQueue.poll()) != null) {

            if (visiting instanceof Probabilistic) {
                if (visiting.isObserved()) {
                    graphBuilder.createConstant(visiting);
                } else {
                    graphBuilder.createVariable(visiting);
                }
            } else {
                graphBuilder.convert(visiting);
            }

        }

        return graphBuilder.build();
    }
}
