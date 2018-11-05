package io.improbable.keanu.codegen.python;

import java.util.Map;

class DocString {
    private String comment;
    private Map<String, String> params;

    DocString(String comment, Map<String, String> params) {
        this.comment = comment;
        this.params = params;
    }

    private boolean isEmpty() {
        return comment.isEmpty() && params.size() == 0;
    }

    String getAsString() {
        if (isEmpty()) {
            return "";
        }
        if (params.size() == 0) {
            return "\"\"\"\n" + comment + "\n\"\"\"\n";
        }
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("\"\"\"\n    ");
        stringBuilder.append(comment.replaceAll("\n ", "\n    "));
        stringBuilder.append("\n");
        for (String param : params.keySet()) {
            stringBuilder.append("\n    ");
            stringBuilder.append(":param ");
            stringBuilder.append(param);
            stringBuilder.append(": ");
            stringBuilder.append(params.get(param));
        }
        stringBuilder.append("\n    \"\"\"\n    ");
        return stringBuilder.toString();
    }
}
