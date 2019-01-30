package io.improbable.keanu.codegen.python.templateobject;

import lombok.Getter;

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

    public VertexConstructor(String javaClass, String pythonClass, String pythonVertexClass, String pythonTypedParameters, String pythonParameters, String docString) {
        this.javaClass = javaClass;
        this.pythonClass = pythonClass;
        this.pythonVertexClass = pythonVertexClass;
        this.pythonTypedParameters = pythonTypedParameters;
        this.pythonParameters = pythonParameters;
        this.docString = docString;
    }

}
