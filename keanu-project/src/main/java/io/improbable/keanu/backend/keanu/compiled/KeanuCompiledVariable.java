package io.improbable.keanu.backend.keanu.compiled;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@AllArgsConstructor
public class KeanuCompiledVariable {

    @Getter
    /**
     * the name of the variable in the source
     */
    private final String name;

    @Getter
    @Setter
    /**
     * True if the variable is not a constant, input or output
     */
    private boolean mutable;
}
