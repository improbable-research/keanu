package io.improbable.keanu.codegen.python;

import io.improbable.keanu.codegen.python.vertex.VertexProcessor;

import java.io.IOException;

public class Runner {

    public static void main(String[] args) throws IOException {
        String generatedDir = args.length == 0 ? "/keanu-python/keanu/vertex/" : args[0];
        VertexProcessor.process(generatedDir);
    }
}