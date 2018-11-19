package io.improbable.keanu.codegen.python;

import java.io.IOException;

public class Runner {

    public static void main(String[] args) throws IOException {
        String generatedDir = args[0];
        VertexProcessor.process(generatedDir);
    }
}