package io.improbable.keanu.codegen.python;

public class Runner {

    public static void main(String[] args) {
        String generatedDir = args[0];
        VertexProcessor.process(generatedDir);
    }
}