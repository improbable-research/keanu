package io.improbable.keanu.codegen.python.datamodel;

import lombok.Getter;
import lombok.Value;

@Value
public class Import {
    @Getter
    private String packageName;
}
