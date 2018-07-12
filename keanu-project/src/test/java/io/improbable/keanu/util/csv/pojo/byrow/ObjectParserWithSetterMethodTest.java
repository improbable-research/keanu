package io.improbable.keanu.util.csv.pojo.byrow;

import io.improbable.keanu.util.csv.pojo.CsvProperty;
import io.improbable.keanu.util.csv.pojo.byrow.RowsAsObjectParser;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static java.util.stream.Collectors.toList;
import static org.junit.Assert.assertEquals;

public class ObjectParserWithSetterMethodTest {

    List<List<String>> data;

    @Before
    public void setup() {
        data = Arrays.asList(
            Arrays.asList("5", "A", "0.11", "0.1", "true"),
            Arrays.asList("12", "6", "0.22", "0.2", "false"),
            Arrays.asList("04", "L", "0.33", "0.3", "true")
        );
    }

    @Test
    public void givenExactNameMatchThenResultIsLoadedCorrectly() {
        List<String> titles = Arrays.asList("myInt", "myString", "myDouble", "myFloat", "myBoolean");

        List<TestPOJOWithSettersClass> output = RowsAsObjectParser.stream(
            TestPOJOWithSettersClass.class,
            data.stream(),
            titles
        ).collect(toList());

        assertEquals(output, Arrays.asList(
            new TestPOJOWithSettersClass(5, "A", 0.11d, 0.1f, true),
            new TestPOJOWithSettersClass(12, "6", 0.22d, 0.2f, false),
            new TestPOJOWithSettersClass(4, "L", 0.33d, 0.3f, true)
        ));
    }

    public static class TestPOJOWithSettersClass {
        private String myString;
        private int myInt;

        public Double myDouble;

        @CsvProperty("myFloat")
        public float someFloat;

        private boolean myBoolean;

        public TestPOJOWithSettersClass() {
        }

        public TestPOJOWithSettersClass(int myInt, String myString, Double myDouble, float someFloat, boolean myBoolean) {
            this.myString = myString;
            this.myInt = myInt;
            this.myDouble = myDouble;
            this.someFloat = someFloat;
            this.myBoolean = myBoolean;
        }

        public void setMyInt(int myInt) {
            this.myInt = myInt;
        }

        public void setMyString(String myString) {
            this.myString = myString;
        }

        @CsvProperty("myBoolean")
        public void loadABoolean(boolean someBool) {
            this.myBoolean = someBool;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            TestPOJOWithSettersClass that = (TestPOJOWithSettersClass) o;

            if (myInt != that.myInt) return false;
            if (myBoolean != that.myBoolean) return false;
            if (Float.compare(that.someFloat, someFloat) != 0) return false;
            if (!myString.equals(that.myString)) return false;
            return myDouble.equals(that.myDouble);
        }
    }
}
