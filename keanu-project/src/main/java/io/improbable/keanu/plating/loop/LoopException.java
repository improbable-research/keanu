package io.improbable.keanu.plating.loop;

public class LoopException extends Exception {
    public LoopException(String message) {
        super(message);
    }

    public LoopException(String message, Throwable cause) {
        super(message, cause);
    }
}
