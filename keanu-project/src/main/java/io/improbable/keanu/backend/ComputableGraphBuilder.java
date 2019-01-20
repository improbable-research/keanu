package io.improbable.keanu.backend;

import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;

import java.util.*;
import java.util.stream.Collectors;

public interface ComputableGraphBuilder<T extends ComputableGraph> {

    void createConstant(Vertex visiting);

    void createVariable(Vertex visiting);

    void create(Vertex visiting);

    void connect(Map<? extends Vertex<?>, ? extends Vertex<?>> connections);

    void registerOutput(VariableReference output);

    Collection<VariableReference> getLatentVariables();

    VariableReference add(VariableReference left, VariableReference right);

    T build();

    default void convert(Collection<? extends Vertex> vertices, Collection<? extends Vertex> outputs) {

        Set<Vertex> outputsUpstreamLambdaSection = outputs.stream()
            .flatMap(
                output -> LambdaSection
                    .getUpstreamLambdaSection(output, true)
                    .getAllVertices().stream()
            )
            .collect(Collectors.toSet());

        List<? extends Vertex> requiredVertices = vertices.stream()
            .filter(outputsUpstreamLambdaSection::contains)
            .collect(Collectors.toList());

        convert(requiredVertices);
        outputs.stream().map(v -> ((Vertex) v).getReference()).forEach(this::registerOutput);
    }

    default void convert(Collection<? extends Vertex> vertices) {

        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(
            Comparator.comparing(Vertex::getId, Comparator.naturalOrder())
        );

        priorityQueue.addAll(vertices);

        Vertex visiting;
        while ((visiting = priorityQueue.poll()) != null) {

            if (visiting instanceof LogProbGraph.PlaceholderVertex) {
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
