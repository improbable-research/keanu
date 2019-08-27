package io.improbable.keanu.vertices.tensor.generic.nonprobabilistic;

import com.google.common.base.Preconditions;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.generic.GenericVertex;

import java.io.PrintStream;

public class PrintVertex<T> extends VertexImpl<T, GenericVertex<T>> implements GenericVertex<T>, NonProbabilistic<T> {

    public static void setPrintStream(PrintStream printStream) {
        PrintVertex.printStream = Preconditions.checkNotNull(printStream);
    }

    private static PrintStream printStream = System.out;

    private static final String PARENT = "parent";

    private static final String MESSAGE = "message";

    private static final String PRINT_DATA = "printData";

    private final Vertex<T, ?> parent;

    private final String message;

    private final boolean printData;

    @ExportVertexToPythonBindings
    public PrintVertex(@LoadVertexParam(PARENT) Vertex<T, ?> parent,
                       @LoadVertexParam(MESSAGE) final String message,
                       @LoadVertexParam(PRINT_DATA) boolean printData) {
        super(parent.getShape());
        this.parent = parent;
        this.message = Preconditions.checkNotNull(message);
        this.printData = printData;
        setParents(parent);
    }

    public PrintVertex(Vertex<T, ?> parent) {
        this(parent, "Calculated Vertex:\n", true);
    }

    @Override
    public T calculate() {
        return print(parent.getValue(), message, printData);
    }

    public static <T> T print(T parentValue, String message, boolean printData) {
        final String dataOutput = printData ? parentValue.toString() + "\n" : "";
        printStream.print(message + dataOutput);
        return parentValue;
    }

    @SaveVertexParam(PARENT)
    public Vertex<T, ?> getParent() {
        return parent;
    }

    @SaveVertexParam(MESSAGE)
    public String getMessage() {
        return message;
    }

    @SaveVertexParam(PRINT_DATA)
    public boolean getPrintData() {
        return printData;
    }
}
