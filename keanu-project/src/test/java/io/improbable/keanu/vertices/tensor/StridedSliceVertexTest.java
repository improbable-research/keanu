package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers;
import org.junit.Test;

public class StridedSliceVertexTest {

    @Test
    public void doesOperateOnMatrix() {

        Slicer slicer = Slicer.builder()
            .all()
            .slice(1)
            .build();

        UnaryOperationTestHelpers.operatesOnInput(tensor -> tensor.slice(slicer), vertex -> vertex.slice(slicer));
    }
}
