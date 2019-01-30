package io.improbable.keanu.codegen.python;

import com.google.common.base.CaseFormat;
import lombok.Getter;

public class PythonParam {
    @Getter
    private String name;
    @Getter
    private Class klass;
    @Getter
    private String defaultValue;

    PythonParam(String name, Class klass, String defaultValue) {
        this.name = CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, name);
        this.klass = klass;
        this.defaultValue = defaultValue;
    }

    PythonParam(String name, Class klass) {
        this(name, klass, "");
    }
}
