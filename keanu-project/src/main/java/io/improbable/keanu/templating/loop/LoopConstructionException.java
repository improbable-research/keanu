package io.improbable.keanu.templating.loop;

public class LoopConstructionException extends RuntimeException {

    public LoopConstructionException(String message) {
        super(message);
    }

    public LoopConstructionException(String message, Throwable cause) {
        super(message, cause);
    }
}
