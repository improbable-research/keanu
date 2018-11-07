package io.improbable.keanu.codegen.python;

import com.sun.javadoc.ConstructorDoc;
import com.sun.javadoc.Tag;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.codegen.python.ParamStringProcessor.getNameToCommentMapping;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.collection.IsCollectionWithSize.hasSize;
import static org.hamcrest.core.IsCollectionContaining.hasItems;
import static org.hamcrest.core.IsEqual.equalTo;
import static org.mockito.Mockito.when;

public class ParamStringProcessorTest {

    private static final String FIRST_PARAM_NAME_CAMEL_CASE = "paramTest";
    private static final String FIRST_PARAM_NAME_SNAKE_CASE = "param_test";
    private static final String SECOND_PARAM_NAME_CAMEL_CASE = "anotherParamTest";
    private static final String SECOND_PARAM_NAME_SNAKE_CASE = "another_param_test";
    private static final String FIRST_PARAM_COMMENT = "This is a comment about the first parameter paramTest";
    private static final String SECOND_PARAM_COMMENT = "This is another comment about the second parameter anotherParamTest";
    private static final String FIRST_PARAM_TEXT = FIRST_PARAM_NAME_CAMEL_CASE + " " + FIRST_PARAM_COMMENT;
    private static final String SECOND_PARAM_TEXT = SECOND_PARAM_NAME_CAMEL_CASE + "   " + SECOND_PARAM_COMMENT;
    private static final String RETURN_TAG_TEXT = "this";

    private static final String PARAM_TAG_NAME = "@param";
    private static final String RETURN_TAG_NAME = "@return";

    @Mock ConstructorDoc constructorDoc;
    @Mock Tag tag1;
    @Mock Tag tag2;
    @Mock Tag tag3;

    @Rule
    public MockitoRule mockitoRule = MockitoJUnit.rule();

    @Before
    public void initialiseMocks() {
        when(constructorDoc.tags(PARAM_TAG_NAME)).thenReturn(new Tag[] {tag1, tag2});

        when(tag1.name()).thenReturn(PARAM_TAG_NAME);
        when(tag1.text()).thenReturn(FIRST_PARAM_TEXT);

        when(tag2.name()).thenReturn(PARAM_TAG_NAME);
        when(tag2.text()).thenReturn(SECOND_PARAM_TEXT);

        when(tag3.name()).thenReturn(RETURN_TAG_NAME);
        when(tag3.text()).thenReturn(RETURN_TAG_TEXT);
    }

    @Test
    public void testParamStringMapping() {
        Map<String, String> nameToCommentMap = getNameToCommentMapping(constructorDoc);
        Set<String> keySet = nameToCommentMap.keySet();
        assertThat(keySet, hasItems(FIRST_PARAM_NAME_SNAKE_CASE, SECOND_PARAM_NAME_SNAKE_CASE));
        assertThat(nameToCommentMap.get(FIRST_PARAM_NAME_SNAKE_CASE), equalTo(FIRST_PARAM_COMMENT));
        assertThat(nameToCommentMap.get(SECOND_PARAM_NAME_SNAKE_CASE), equalTo(SECOND_PARAM_COMMENT));
        assertThat(keySet, hasSize(2));
    }
}
