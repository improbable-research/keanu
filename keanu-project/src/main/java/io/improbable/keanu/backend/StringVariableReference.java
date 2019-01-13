package io.improbable.keanu.backend;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;

@AllArgsConstructor
@EqualsAndHashCode
public class StringVariableReference implements VariableReference {

    private final String reference;

    @Override
    public String toStringReference() {
        return reference;
    }
}
