package io.improbable.keanu.codegen.python;

import java.util.Map;

class DocString {
    private String comment;
    private Map<String, String> params;
    private String methodName;

    DocString(String comment, Map<String, String> params, String methodName) {
        this.comment = comment;
        this.params = params;
        this.methodName = methodName;
    }

    String getComment() {
        return comment;
    }

    Map<String, String> getParams() {
        return params;
    }

    String getMethodName() {
        return methodName;
    }
}
