package io.improbable.keanu.codegen.python;

import com.sun.javadoc.ConstructorDoc;

import java.util.HashMap;
import java.util.Map;

import static com.google.common.base.CaseFormat.LOWER_UNDERSCORE;
import static com.google.common.base.CaseFormat.UPPER_CAMEL;

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
            String snakeCaseParamName = UPPER_CAMEL.to(LOWER_UNDERSCORE, splitComment[2]);
            String paramComment = splitComment[3];
            nameToCommentMapping.put(snakeCaseParamName, paramComment);
        }
        return nameToCommentMapping;
    }
}
