package io.improbable.keanu.util.csv.pojo.bycolumn;

import io.improbable.keanu.util.csv.pojo.PublicFieldMatcher;
import io.improbable.keanu.util.csv.pojo.SetterMatcher;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ColumnsVectorizedObjectParser<T> {

    private static final boolean IGNORE_MISSING_FIELDS_DEFAULT = false;

    private final Class<T> base;
    private final Stream<List<String>> inputStream;
    private final List<String> csvTitles;

    public ColumnsVectorizedObjectParser(Class<T> base, Stream<List<String>> inputStream, List<String> csvTitles) {
        this.base = base;
        this.inputStream = inputStream;
        this.csvTitles = csvTitles;
    }

    public T load() {
        return load(IGNORE_MISSING_FIELDS_DEFAULT);
    }

    public T load(boolean ignoreUnmatchedFields) {

        List<List<String>> csvColumns = csvTitles.stream()
            .map(title -> new ArrayList<String>())
            .collect(Collectors.toList());

        inputStream.forEach(line -> {
            for (int i = 0; i < line.size(); i++) {
                csvColumns.get(i).add(line.get(i));
            }
        });

        List<CsvColumnConsumer<T>> columnConsumers = getColumnConsumers(base, csvTitles, ignoreUnmatchedFields);

        return deserialize(csvColumns, columnConsumers, base);
    }

    private static <T> List<CsvColumnConsumer<T>> getColumnConsumers(Class<T> base,
                                                                     List<String> fieldTitles,
                                                                     boolean ignoreUnmatchedFields) {

        List<CsvColumnConsumer<T>> columnConsumers = new ArrayList<>();
        CsvColumnConsumer<T> defaultConsumer = (target, value) -> {
        };

        List<Field> potentialFields = new ArrayList<>(Arrays.asList(base.getFields()));
        List<Method> potentialSetters = new ArrayList<>(Arrays.asList(base.getMethods()));

        for (String title : fieldTitles) {

            Optional<CsvColumnConsumer<T>> consumerForTitle;

            consumerForTitle = PublicFieldMatcher.getFieldColumnConsumer(title, potentialFields);

            if (!consumerForTitle.isPresent()) {
                consumerForTitle = SetterMatcher.getSetterColumnConsumer(title, potentialSetters);
            }

            if (!consumerForTitle.isPresent() && !ignoreUnmatchedFields) {
                throw new IllegalArgumentException("Unable to find field for csv data \"" + title + "\"");
            }

            columnConsumers.add(consumerForTitle.orElse(defaultConsumer));
        }
        return columnConsumers;
    }

    private static <T> T deserialize(List<List<String>> csvColumns,
                                     List<CsvColumnConsumer<T>> fieldMappers,
                                     Class<T> base) {

        try {
            T target = base.newInstance();

            for (int i = 0; i < csvColumns.size(); i++) {
                BiConsumer<T, List<String>> fieldConsumer = fieldMappers.get(i);
                if (fieldConsumer != null) {
                    fieldConsumer.accept(target, csvColumns.get(i));
                }
            }

            return target;
        } catch (IllegalAccessException | InstantiationException e) {
            throw new IllegalStateException(e);
        }
    }
}
