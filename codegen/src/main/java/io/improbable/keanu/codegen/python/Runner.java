package io.improbable.keanu.codegen.python;

import freemarker.template.TemplateException;

import java.io.IOException;

public class Runner {

    public static void main(String[] args) throws IOException, TemplateException {
        String generatedDir = args[0];
        VertexProcessor.process(generatedDir);
    }
}