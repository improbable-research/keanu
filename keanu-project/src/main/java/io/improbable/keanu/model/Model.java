package io.improbable.keanu.model;

public interface Model<INPUT, OUTPUT> {

    OUTPUT predict(INPUT input);

}
