package io.improbable.keanu.model;

public interface PredictiveModel<INPUT, OUTPUT> {

    OUTPUT predict(INPUT input);

}
