package io.improbable.keanu.plating.loop;

public class LoopConstructionException extends RuntimeException {

  public LoopConstructionException(String message) {
    super(message);
  }

  public LoopConstructionException(String message, Throwable cause) {
    super(message, cause);
  }
}
