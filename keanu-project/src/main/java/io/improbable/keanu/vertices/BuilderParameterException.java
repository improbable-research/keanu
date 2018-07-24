package io.improbable.keanu.vertices;

public class BuilderParameterException extends IllegalStateException {
    public BuilderParameterException(String s) {
        super(s);
    }

    public BuilderParameterException(String message, Throwable cause) {
        super(message, cause);
    }
}
