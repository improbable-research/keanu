package io.improbable.keanu.codegen.python;

import com.sun.javadoc.ConstructorDoc;
import com.sun.javadoc.Parameter;
import com.sun.javadoc.Tag;

import java.util.HashMap;
import java.util.Map;

import static com.google.common.base.CaseFormat.LOWER_UNDERSCORE;
import static com.google.common.base.CaseFormat.UPPER_CAMEL;

class ParamStringProcessor {
    static Map<String, String> getNameToCommentMapping(ConstructorDoc constructorDoc) {
        Map<String, String> nameToCommentMapping = new HashMap<>();

        for(Parameter parameter: constructorDoc.parameters()) {
            String name = UPPER_CAMEL.to(LOWER_UNDERSCORE, parameter.name());
            String description = "";
            nameToCommentMapping.put(name, description);
        }

        Tag[] params = constructorDoc.tags("@param");
        for (Tag param: params) {
            String[] text = param.text().split(" ", 2);
            String snakeCaseParamName = UPPER_CAMEL.to(LOWER_UNDERSCORE, text[0]);
            String paramComment = text[1].trim();
            nameToCommentMapping.put(snakeCaseParamName, paramComment);
        }
        return nameToCommentMapping;
    }
}
