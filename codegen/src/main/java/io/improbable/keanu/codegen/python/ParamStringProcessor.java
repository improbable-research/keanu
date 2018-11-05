package io.improbable.keanu.codegen.python;

import com.sun.javadoc.ConstructorDoc;

import java.util.HashMap;
import java.util.Map;

class ParamStringProcessor {
    static Map<String, String> getNameToCommentMapping(ConstructorDoc constructorDoc) {
        String rawComment = constructorDoc.getRawCommentText();
        String[] rawCommentLines = rawComment.split("\\r?\\n");
        Map<String, String> nameToCommentMapping = new HashMap<>();
        for (String commentLine : rawCommentLines) {
            if (!commentLine.contains("@param")) {
                continue;
            }
            commentLine = commentLine.replaceFirst("[ ]{2,}", " ");
            String[] splitComment = commentLine.split(" ", 4);
            String snakeCaseParamName = toSnakeCase(splitComment[2]);
            String paramComment = splitComment[3];
            nameToCommentMapping.put(snakeCaseParamName, paramComment);
        }
        return nameToCommentMapping;
    }

    static String toSnakeCase(String camelCase) {
        String[] camelCaseWords = camelCase.split("(?=\\p{Upper})");
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(camelCaseWords[0]);
        for(int i=1; i<camelCaseWords.length; i++) {
            stringBuilder.append("_");
            stringBuilder.append(camelCaseWords[i].substring(0,1).toLowerCase());
            stringBuilder.append(camelCaseWords[i].substring(1));
        }
        return stringBuilder.toString();
    }
}
