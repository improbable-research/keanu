package io.improbable.keanu.model;

public interface ModelFitter<INPUT, OUTPUT> {
    void fit(ModelGraph<INPUT, OUTPUT> modelGraph, INPUT input, OUTPUT output);
}
