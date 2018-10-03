package io.improbable.keanu.util;

import lombok.AllArgsConstructor;
import lombok.Value;

@Value
@AllArgsConstructor
public class ProgressUpdate {
  private final String message;
  private final Double progressPercentage;

  public ProgressUpdate() {
    this.message = null;
    this.progressPercentage = null;
  }

  public ProgressUpdate(String message) {
    this.message = message;
    this.progressPercentage = null;
  }

  public ProgressUpdate(Double progressPercentage) {
    this.message = null;
    this.progressPercentage = progressPercentage;
  }
}
