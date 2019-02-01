package io.improbable.keanu.codegen.python.datamodel;

import lombok.Getter;
import lombok.Value;

@Value
public class VertexConstructor {

    @Getter
    private String javaClass;
    @Getter
    private String pythonClass;
    @Getter
    private String pythonVertexClass;
    @Getter
    private String pythonTypedParameters;
    @Getter
    private String pythonParameters;
    @Getter
    private String docString;

}
