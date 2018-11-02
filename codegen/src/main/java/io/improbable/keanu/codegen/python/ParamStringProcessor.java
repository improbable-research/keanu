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
            nameToCommentMapping.put(splitComment[2], splitComment[3]);
        }
        return nameToCommentMapping;
    }
}
