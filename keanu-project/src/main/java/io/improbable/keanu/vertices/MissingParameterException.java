package io.improbable.keanu.vertices;

public class MissingParameterException extends IllegalStateException {
    public MissingParameterException(String s) {
        super(s);
    }

    public MissingParameterException(String message, Throwable cause) {
        super(message, cause);
    }
}
