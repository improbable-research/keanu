package io.improbable.keanu.plating.loop;

public class LoopDidNotTerminateException extends RuntimeException {
    public LoopDidNotTerminateException(String message) {
        super(message);
    }

    public LoopDidNotTerminateException(String message, Throwable cause) {
        super(message, cause);
    }
}
