package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Probabilistic;
import org.junit.Test;
import org.reflections.Reflections;

import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static io.improbable.keanu.backend.keanu.compiled.KeanuVertexToTensorOpMapper.getOpMapperFor;

public class KeanuVertexToTensorOpMapperTest {

    @Test
    public void doesSupportAllOperations() {

        Reflections reflections = new Reflections("io.improbable.keanu.vertices");

        Set<Class<? extends IVertex>> vertices = reflections.getSubTypesOf(IVertex.class);
        List<Class<? extends IVertex>> supportedVertices = vertices.stream()
            .filter(v -> !Probabilistic.class.isAssignableFrom(v))
            .filter(v -> !NonSaveableVertex.class.isAssignableFrom(v))
            .filter(v -> !Modifier.isAbstract(v.getModifiers()))
            .collect(Collectors.toList());

        assertDoesSupport(supportedVertices);
    }

    private void assertDoesSupport(List<Class<? extends IVertex>> supportedVertices) {
        List<Class<? extends IVertex>> unsupportedVertices = supportedVertices.stream()
            .filter(v -> getOpMapperFor(v) == null)
            .collect(Collectors.toList());

        if (!unsupportedVertices.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            sb.append("\n");
            unsupportedVertices.forEach(v -> sb.append("Does not support " + v.getCanonicalName() + "\n"));
            throw new AssertionError(sb.toString());
        }
    }

}
