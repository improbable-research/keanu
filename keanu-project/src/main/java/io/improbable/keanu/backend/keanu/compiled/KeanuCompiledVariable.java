package io.improbable.keanu.backend.keanu.compiled;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@AllArgsConstructor
public class KeanuCompiledVariable {
    @Getter
    private final String name;

    @Getter
    @Setter
    private boolean mutable;
}
