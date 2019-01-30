package io.improbable.keanu.codegen.python;

import com.google.common.collect.ImmutableMap;
import com.sun.javadoc.ConstructorDoc;
import com.sun.javadoc.Tag;
import lombok.experimental.UtilityClass;

import java.util.Map;

import static com.google.common.base.CaseFormat.LOWER_UNDERSCORE;
import static com.google.common.base.CaseFormat.UPPER_CAMEL;

@UtilityClass
class ParamStringProcessor {
    Map<String, String> getNameToCommentMapping(ConstructorDoc constructorDoc) {
        ImmutableMap.Builder<String, String> nameToCommentMapping = ImmutableMap.builder();
        Tag[] params = constructorDoc.tags("@param");
        for (Tag param: params) {
            String[] text = param.text().split(" ", 2);
            String snakeCaseParamName = UPPER_CAMEL.to(LOWER_UNDERSCORE, text[0]);
            String paramComment = text[1].trim();
            nameToCommentMapping.put(snakeCaseParamName, paramComment);
        }
        return nameToCommentMapping.build();
    }
}
