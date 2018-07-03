## Reading data into Keanu

### CSV data

Keanu is packaged with a csv reader that allows you to load simple csv files with or without
a header. You can read from a specified file, predefined string or a file on the class path.
To do this, first create a CsvReader and then get the lines from the reader.

#### Creating a CsvReader from your csv data
To read from a file from your current directory:

```java
CsvReader reader = ReadCsv.fromFile("~/my_filename.csv");

```

If you place the csv file in your src/main/resources folder, you can load it as a resource,
which you would do in order to avoid having to provide a hardcoded file path.

```java
CsvReader reader = ReadCsv.fromResources("my_filename.csv");

```

Once you have a CsvReader, you can call readLines() to start reading each line.
```java
for (List<String> csvLine : csvReader.readLines()) {
    //do something with your csv line
}
```

#### Large CSV files

If you csv is very large, you may not want to load the entire csv file into memory before
processing it. You can stream the lines in order to avoid holding the entire file in memory.

Once you have a CsvReader, you can call streamLines() to start streaming each line. Make sure
to close the stream as it is potentially connected to an open file. Closing the stream can be
done by using try-with-resources, which is shown below.

```java
try (Stream<List<String>> lineStream = csvReader.streamLines()) {
   
    lineStream.forEach(line -> {
       //do something with your line. 
    });
}
```

### Reading CSV as a plain old java object (POJO)

If you don't feel like processing the raw lines from your CSV file, then you 
have the option to read directly to a Java object.

Given a POJO

```java
public class MyClass {
    public String myString;
    public int myInt;
}
```

And some CSV

```
myString,myInt
aString,1
bString,2
cString,3
```

The CSV can be loaded as a Java object by

```java
List<MyClass> myPojos = ReadCsv.fromFile("some/file/path")
                               .asRowsDefinedBy(MyClass.class)
                               .load();
```

If your csv header names contain illegal characters you have the option to
tag the field with a Java annotation.

Given CSV

```
my-problematic*header,myInt
aString,1
bString,2
cString,3
```

Annotate your class with `@CsvProperty` and the problematic header title.

```java
public class MyOtherClass {
    @CsvProperty("my-problematic*header")
    public String myString;
    public int myInt;
}
```

The CSV parser also supports using setter methods.

```java
public class MyClass {
    private String myString;
    
    public void setMyString(String aString){
        ...
    }
}
```