package io.improbable.keanu.codegen.python;

import lombok.Getter;

/**
 * Encapsulates data required for defining a python parameter
 */
public class PythonParam {
    @Getter
    private String name;
    @Getter
    private Class klass;
    @Getter
    private String defaultValue;

    public PythonParam(String name, Class klass, String defaultValue) {
        this.name = ParamStringProcessor.toLowerUnderscore(name);
        this.klass = klass;
        this.defaultValue = defaultValue;
    }

    public PythonParam(String name, Class klass) {
        this(name, klass, null);
    }

    public boolean hasDefaultValue() {
        return defaultValue != null;
    }
}
