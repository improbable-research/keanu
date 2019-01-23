package io.improbable.keanu.codegen.python;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.vertices.Vertex;
import org.reflections.Reflections;

import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.util.Set;

public class GetNonExportedVertices {

    public static void main(String[] args) {
        Reflections reflections = new Reflections("io.improbable.keanu.vertices");

        Set<Class<? extends Vertex>> vertices = reflections.getSubTypesOf(Vertex.class);

        for (Class vertexClass : vertices) {
            if (!Modifier.isAbstract(vertexClass.getModifiers()) && vertexNotExportedToPython(vertexClass)) {
                System.out.println(vertexClass.getSimpleName());
            }
        }
    }

    private static boolean vertexNotExportedToPython(Class vertexClass) {
        for (Constructor constructor : vertexClass.getConstructors()) {
            if (constructor.isAnnotationPresent(ExportVertexToPythonBindings.class)) {
                return false;
            }
        }

        return true;
    }
}
