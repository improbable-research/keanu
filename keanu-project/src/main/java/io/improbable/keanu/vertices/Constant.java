package io.improbable.keanu.vertices;

/**
 * This interface is used to identify vertices that are constants.
 *
 * @param <T> constant type
 */
public interface Constant<T> {

    T getValue();
}
