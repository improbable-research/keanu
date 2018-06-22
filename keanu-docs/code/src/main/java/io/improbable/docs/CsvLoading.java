package io.improbable.docs;

import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.util.csv.pojo.CsvProperty;

import java.util.List;

public class CsvLoading {

    public static void main(String[] args) {

        List<MyClass> myPojos = ReadCsv.fromFile("some/file/path")
            .asRowsDefinedBy(MyClass.class)
            .load();
    }

    public class MyClass {
        public String myString;
        public int myInt;
    }

    public class MyOtherClass {
        @CsvProperty("my-problematic*header")
        public String myString;
        public int myInt;
    }
}
