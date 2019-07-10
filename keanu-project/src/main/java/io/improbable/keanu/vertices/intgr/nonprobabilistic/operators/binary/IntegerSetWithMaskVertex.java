package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerBinaryOpVertex;

public class IntegerSetWithMaskVertex extends IntegerBinaryOpVertex {

    private static final String REPLACE_WITH_VALUE = "replaceWithValue";
    private final Integer replaceWithValue;

    @ExportVertexToPythonBindings
    public IntegerSetWithMaskVertex(@LoadVertexParam(LEFT_NAME) IntegerVertex left,
                                    @LoadVertexParam(RIGHT_NAME) IntegerVertex mask,
                                    @LoadVertexParam(REPLACE_WITH_VALUE) Integer replaceWithValue) {
        super(left, mask);
        this.replaceWithValue = replaceWithValue;
    }

    @Override
    protected IntegerTensor op(IntegerTensor l, IntegerTensor mask) {
        return l.setWithMask(mask, replaceWithValue);
    }

    @SaveVertexParam(REPLACE_WITH_VALUE)
    public Integer getReplaceWithValue() {
        return replaceWithValue;
    }
}
