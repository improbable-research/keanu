package io.improbable.keanu.vertices.utility;

import com.google.common.base.Preconditions;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import java.io.PrintStream;

public class PrintVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor>, NonSaveableVertex {
    private PrintStream printStream = System.out;
    private final DoubleVertex parent;

    @ExportVertexToPythonBindings
    public PrintVertex(DoubleVertex parent, final PrintStream printStream) {
        this(parent);
        Preconditions.checkNotNull(printStream);
        this.printStream = printStream;
    }

    @ExportVertexToPythonBindings
    public PrintVertex(DoubleVertex parent) {
        super(parent.getShape());
        this.parent = parent;
        setParents(parent);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return parent.sample();
    }

    @Override
    public DoubleTensor calculate() {
        final String label = parent.getLabel() == null ? "<no label>" : parent.getLabel().getUnqualifiedName();
        final String statement = String.format("Calculated Vertex: (label: %s, data: %s)", label,  parent.getValue());
        printStream.println(statement);
        return parent.getValue();
    }
}
