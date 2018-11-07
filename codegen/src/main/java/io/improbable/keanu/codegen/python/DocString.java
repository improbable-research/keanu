package io.improbable.keanu.codegen.python;

import java.util.Map;

class DocString {
    private static final String THREE_QUOTES = "\"\"\"";
    private static final String NEW_LINE_TAB = "\n    ";

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
            return THREE_QUOTES + "\n" + comment + THREE_QUOTES + "\n\n";
        }
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(THREE_QUOTES);
        stringBuilder.append(NEW_LINE_TAB);
        stringBuilder.append(comment.replaceAll("\n ", NEW_LINE_TAB));
        stringBuilder.append(NEW_LINE_TAB);
        for (String param : params.keySet()) {
            stringBuilder.append(NEW_LINE_TAB);
            stringBuilder.append(":param ");
            stringBuilder.append(param);
            stringBuilder.append(": ");
            stringBuilder.append(params.get(param));
        }
        stringBuilder.append(NEW_LINE_TAB);
        stringBuilder.append(THREE_QUOTES);
        return stringBuilder.toString();
    }
}
