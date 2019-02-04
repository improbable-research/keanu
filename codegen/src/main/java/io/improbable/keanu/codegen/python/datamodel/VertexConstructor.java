package io.improbable.keanu.codegen.python.datamodel;

import lombok.Value;

@Value
public class VertexConstructor {

    private String javaClass;
    private String pythonClass;
    private String pythonVertexClass;
    private String pythonTypedParameters;
    private String pythonParameters;
    private String docString;

}
