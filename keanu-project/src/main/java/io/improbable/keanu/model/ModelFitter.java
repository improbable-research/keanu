package io.improbable.keanu.model;

public interface ModelFitter<INPUT, OUTPUT> {
    void fit(INPUT input, OUTPUT output);
}
