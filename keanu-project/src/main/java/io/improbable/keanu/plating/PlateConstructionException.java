package io.improbable.keanu.plating;

/**
 * This extends RuntimeException because Plates are typically used inside a lambda function And
 * checked exceptions in a lambda must be caught, which leads to ugly code See
 * https://stackoverflow.com/a/27668305/741789
 */
public class PlateConstructionException extends RuntimeException {

    public PlateConstructionException(String message) {
        super(message);
    }

    public PlateConstructionException(String message, Throwable cause) {
        super(message, cause);
    }
}
