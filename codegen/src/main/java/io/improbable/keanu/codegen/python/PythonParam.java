package io.improbable.keanu.codegen.python;

import lombok.Getter;

import java.util.Objects;

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

    @Override
    public boolean equals(Object other) {
        if (this == other) return true;
        if (other == null || getClass() != other.getClass()) return false;
        PythonParam that = (PythonParam) other;
        return Objects.equals(name, that.name);
    }
}
