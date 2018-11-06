package io.improbable.keanu.codegen.python;

import com.sun.javadoc.ConstructorDoc;
import com.sun.javadoc.Tag;

import java.util.HashMap;
import java.util.Map;

import static com.google.common.base.CaseFormat.LOWER_UNDERSCORE;
import static com.google.common.base.CaseFormat.UPPER_CAMEL;

class ParamStringProcessor {
    static Map<String, String> getNameToCommentMapping(ConstructorDoc constructorDoc) {
        Map<String, String> nameToCommentMapping = new HashMap<>();
        Tag[] params = constructorDoc.tags("@param");
        for (Tag param: params) {
            String snakeCaseParamName = UPPER_CAMEL.to(LOWER_UNDERSCORE, param.name());
            String paramComment = param.text();
            nameToCommentMapping.put(snakeCaseParamName, paramComment);
        }
        return nameToCommentMapping;
    }
}
