package io.improbable.keanu.plating;


/**
 * This extends RuntimeException because Sequence are typically used inside a lambda function
 * And checked exceptions in a lambda must be caught, which leads to ugly code
 * See https://stackoverflow.com/a/27668305/741789
 */
public class SequenceConstructionException extends RuntimeException {

    public SequenceConstructionException(String message) {
        super(message);
    }

    public SequenceConstructionException(String message, Throwable cause) {
        super(message, cause);
    }
}
