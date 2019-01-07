package io.improbable.keanu.vertices.utility;

/**
 * {@code GraphAssertionException} is an unchecked exception thrown by
 * an {@link AssertVertex} when its assertion is triggered.
 */
public class GraphAssertionException extends RuntimeException {

    public GraphAssertionException(String message) {
        super(message);
    }

}
