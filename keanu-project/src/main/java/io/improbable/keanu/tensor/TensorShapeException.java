package io.improbable.keanu.tensor;

public class TensorShapeException extends IllegalArgumentException {

    public TensorShapeException(String message) {
        super(message);
    }

    public TensorShapeException(String message, Throwable cause) {
        super(message, cause);
    }
}
