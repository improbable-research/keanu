package io.improbable.keanu.util.csv.pojo.byrow;

import static java.util.stream.Collectors.toList;
import static org.junit.Assert.assertEquals;

import io.improbable.keanu.util.csv.pojo.CsvProperty;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import org.junit.Before;
import org.junit.Test;

public class ObjectParserWithPublicFieldTest {

    List<List<String>> data;

    @Before
    public void setup() {
        data =
                Arrays.asList(
                        Arrays.asList("5", "A"),
                        Arrays.asList("12", "6"),
                        Arrays.asList("04", "L"));
    }

    @Test
    public void givenExactNameMatchThenResultIsLoadedCorrectly() {
        List<String> titles = Arrays.asList("id", "name");

        List<TestPOJOWithPublicFieldsClass> output =
                RowsAsObjectParser.stream(
                                TestPOJOWithPublicFieldsClass.class, data.stream(), titles)
                        .collect(toList());

        assertEquals(
                output,
                Arrays.asList(
                        new TestPOJOWithPublicFieldsClass(5, "A"),
                        new TestPOJOWithPublicFieldsClass(12, "6"),
                        new TestPOJOWithPublicFieldsClass(4, "L")));
    }

    @Test(expected = IllegalArgumentException.class)
    public void givenWrongNamesThenResultIsNotLoaded() {
        List<String> titles = Arrays.asList("abc", "efg");
        RowsAsObjectParser.stream(TestPOJOWithPublicFieldsClass.class, data.stream(), titles);
    }

    @Test(expected = IllegalArgumentException.class)
    public void givenSetToNotIgnoreFieldsThenThrowsExceptionForMissingField() {
        List<String> titles = Arrays.asList("id", "name", "blah");

        RowsAsObjectParser.stream(
                TestPOJOWithPublicFieldsClass.class, data.stream(), titles, false);
    }

    @Test
    public void givenCaseDifferenceThenStillExactMatch() {
        List<String> titles = Arrays.asList("Id", "name");

        List<TestPOJOWithPublicFieldsClass> output =
                RowsAsObjectParser.stream(
                                TestPOJOWithPublicFieldsClass.class, data.stream(), titles)
                        .collect(toList());

        assertEquals(
                output,
                Arrays.asList(
                        new TestPOJOWithPublicFieldsClass(5, "A"),
                        new TestPOJOWithPublicFieldsClass(12, "6"),
                        new TestPOJOWithPublicFieldsClass(4, "L")));
    }

    public static class TestPOJOWithPublicFieldsClass {

        @CsvProperty("name")
        public String myName;

        public int id;

        public TestPOJOWithPublicFieldsClass() {}

        public TestPOJOWithPublicFieldsClass(int id, String name) {
            this.id = id;
            this.myName = name;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            TestPOJOWithPublicFieldsClass that = (TestPOJOWithPublicFieldsClass) o;
            return id == that.id && Objects.equals(myName, that.myName);
        }

        @Override
        public int hashCode() {

            return Objects.hash(myName, id);
        }
    }
}
