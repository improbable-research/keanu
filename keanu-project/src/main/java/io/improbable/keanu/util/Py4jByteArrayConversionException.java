package io.improbable.keanu.util;

public class Py4jByteArrayConversionException extends RuntimeException {
    Py4jByteArrayConversionException(String message) {
        super(message);
    }

    Py4jByteArrayConversionException(String message, Throwable cause) {
        super(message, cause);
    }
}
