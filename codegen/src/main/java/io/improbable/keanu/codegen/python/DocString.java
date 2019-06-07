package io.improbable.keanu.codegen.python;

import org.apache.commons.lang3.StringUtils;
import java.util.Map;
import java.util.Map.Entry;

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
        for (Entry<String, String> entry : parameterNameToDescriptionMap.entrySet()) {
            stringBuilder.append(NEW_LINE_TAB);
            stringBuilder.append(":param ");
            stringBuilder.append(entry.getKey());
            stringBuilder.append(": ");
            stringBuilder.append(entry.getValue());
        }
        stringBuilder.append(NEW_LINE_TAB);
        stringBuilder.append(THREE_QUOTES);
        stringBuilder.append(NEW_LINE_TAB);
        return stringBuilder.toString();
    }
}
