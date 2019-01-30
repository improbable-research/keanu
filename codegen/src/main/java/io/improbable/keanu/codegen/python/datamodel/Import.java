package io.improbable.keanu.codegen.python.datamodel;

import lombok.Getter;

public class Import {
    @Getter
    private String packageName;

    public Import(String packageName) {
        this.packageName = packageName;
    }
}
