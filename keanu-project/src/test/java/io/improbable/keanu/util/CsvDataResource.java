package io.improbable.keanu.util;

import io.improbable.keanu.util.csv.ReadCsv;
import org.junit.rules.ExternalResource;

public class CsvDataResource<T> extends ExternalResource {
    private T data;

    public CsvDataResource(String fileLocation, Class<T> clazz) {
        this.data = ReadCsv
            .fromResources(fileLocation)
            .asVectorizedColumnsDefinedBy(clazz)
            .load(true);
    }

    public T getData() {
        return this.data;
    }
}
