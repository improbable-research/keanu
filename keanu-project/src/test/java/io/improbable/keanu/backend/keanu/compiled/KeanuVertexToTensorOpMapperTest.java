package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import org.junit.Test;
import org.reflections.Reflections;

import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static io.improbable.keanu.backend.keanu.compiled.KeanuVertexToTensorOpMapper.getOpMapperFor;

public class KeanuVertexToTensorOpMapperTest {

    @Test
    public void doesSupportIntegerOperations() {

        Reflections reflections = new Reflections("io.improbable.keanu.vertices.intgr.nonprobabilistic");

        Set<Class<? extends Vertex>> vertices = reflections.getSubTypesOf(Vertex.class);
        List<Class<? extends Vertex>> supportedVertices = vertices.stream()
            .filter(v -> !NonSaveableVertex.class.isAssignableFrom(v))
            .filter(v -> !Modifier.isAbstract(v.getModifiers()))
            .collect(Collectors.toList());

        assertDoesSupport(supportedVertices);
    }

    @Test
    public void doesSupportDoubleOperations() {

        Reflections reflections = new Reflections("io.improbable.keanu.vertices.dbl.nonprobabilistic");

        Set<Class<? extends Vertex>> vertices = reflections.getSubTypesOf(Vertex.class);
        List<Class<? extends Vertex>> supportedVertices = vertices.stream()
            .filter(v -> !NonSaveableVertex.class.isAssignableFrom(v))
            .filter(v -> !Modifier.isAbstract(v.getModifiers()))
            .collect(Collectors.toList());

        assertDoesSupport(supportedVertices);
    }

    private void assertDoesSupport(List<Class<? extends Vertex>> supportedVertices) {
        List<Class<? extends Vertex>> unsupportedVertices = supportedVertices.stream()
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
