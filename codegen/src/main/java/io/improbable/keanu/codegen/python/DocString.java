package io.improbable.keanu.codegen.python;

import org.apache.commons.lang3.StringUtils;

import java.util.Map;

public class DocString {
    private static final String THREE_QUOTES = "\"\"\"";
    private static final String NEW_LINE_TAB = "\n    ";

    private final String methodDescription;
    private final Map<String, String> parameterNameToDescriptionMap;

    DocString(String methodDescription, Map<String, String> parameterNameToDescriptionMap) {
        this.methodDescription = methodDescription;
        this.parameterNameToDescriptionMap = parameterNameToDescriptionMap;
    }

    private boolean isEmpty() {
        return StringUtils.isEmpty(methodDescription) && parameterNameToDescriptionMap.isEmpty();
    }

    public String getAsString() {
        if (isEmpty()) {
            return "";
        }
        if (parameterNameToDescriptionMap.size() == 0) {
            return THREE_QUOTES + "\n" + methodDescription + THREE_QUOTES + "\n\n";
        }
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(THREE_QUOTES);
        stringBuilder.append(NEW_LINE_TAB);
        stringBuilder.append(methodDescription.replaceAll("\n ", NEW_LINE_TAB));
        stringBuilder.append(NEW_LINE_TAB);
        for (String param : parameterNameToDescriptionMap.keySet()) {
            stringBuilder.append(NEW_LINE_TAB);
            stringBuilder.append(":param ");
            stringBuilder.append(param);
            stringBuilder.append(": ");
            stringBuilder.append(parameterNameToDescriptionMap.get(param));
        }
        stringBuilder.append(NEW_LINE_TAB);
        stringBuilder.append(THREE_QUOTES);
        stringBuilder.append(NEW_LINE_TAB);
        return stringBuilder.toString();
    }
}
