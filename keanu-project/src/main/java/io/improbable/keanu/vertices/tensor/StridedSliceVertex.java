package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StridedSliceVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private final static String START_NAME = "start";
    private final static String END_NAME = "end";
    private final static String STRIDE_NAME = "stride";
    private final static String ELLIPSIS_NAME = "ellipsis";
    private final static String UPPER_BOUND_STOP_NAME = "upperBoundStop";
    private final static String DROP_DIMENSION_NAME = "dropDimension";

    private final long[] start;
    private final long[] end;
    private final long[] stride;
    private final boolean[] upperBoundStop;
    private final boolean[] dropDimension;
    private final Integer ellipsis;

    private final Slicer slicer;

    public StridedSliceVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                              @LoadVertexParam(START_NAME) long[] start,
                              @LoadVertexParam(END_NAME) long[] end,
                              @LoadVertexParam(STRIDE_NAME) long[] stride,
                              @LoadVertexParam(ELLIPSIS_NAME) Integer ellipsis,
                              @LoadVertexParam(UPPER_BOUND_STOP_NAME) boolean[] upperBoundStop,
                              @LoadVertexParam(DROP_DIMENSION_NAME) boolean[] dropDimension) {
        this(inputVertex, new Slicer(start, end, stride, ellipsis, upperBoundStop, dropDimension));
    }

    @ExportVertexToPythonBindings
    public StridedSliceVertex(TensorVertex<T, TENSOR, VERTEX> inputVertex, Slicer slicer) {
        super(slicer.getResultShape(inputVertex.getShape(), false), inputVertex, inputVertex.ofType());
        this.slicer = slicer;

        List<Slicer.Slice> slices = slicer.getSlices();
        this.ellipsis = slicer.getEllipsisPosition();
        this.start = new long[slices.size()];
        this.end = new long[slices.size()];
        this.stride = new long[slices.size()];
        this.upperBoundStop = new boolean[slices.size()];
        this.dropDimension = new boolean[slices.size()];

        for (int i = 0; i < slices.size(); i++) {
            final Slicer.Slice slice = slices.get(i);
            start[i] = slice.getStart();

            upperBoundStop[i] = slice.isUpperBoundStop();
            if (!upperBoundStop[i]) {
                end[i] = slice.getStop();
            }

            stride[i] = slice.getStep();
            dropDimension[i] = slice.isDropDimension();
        }
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.slice(slicer);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();

        DoubleTensor partial = derivativeOfOutputWithRespectToSelf.get();

        DoubleTensor result = null;

        partials.put(inputVertex, new PartialDerivative(result));

        return partials;
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative dInputVertex = derivativeOfParentsWithRespectToInput.get(inputVertex);

        return new PartialDerivative(dInputVertex.get().slice(slicer));
    }

    @SaveVertexParam(START_NAME)
    public long[] getStart() {
        return this.start;
    }

    @SaveVertexParam(END_NAME)
    public long[] getEnd() {
        return this.end;
    }

    @SaveVertexParam(STRIDE_NAME)
    public long[] getStride() {
        return this.stride;
    }

    @SaveVertexParam(UPPER_BOUND_STOP_NAME)
    public boolean[] getUpperBoundStop() {
        return this.upperBoundStop;
    }

    @SaveVertexParam(DROP_DIMENSION_NAME)
    public boolean[] getDropDimension() {
        return this.dropDimension;
    }

    @SaveVertexParam(ELLIPSIS_NAME)
    public Integer getEllipsis() {
        return this.ellipsis;
    }
}
