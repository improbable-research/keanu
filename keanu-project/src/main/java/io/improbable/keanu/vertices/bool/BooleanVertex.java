package io.improbable.keanu.vertices.bool;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.BaseBooleanTensor;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.kotlin.BooleanOperators;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.TensorVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.AndBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.OrBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.XorBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.AndMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.BooleanConcatenationVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.OrMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanReshapeVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanSliceVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanTakeVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanUnaryOpVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.NotBinaryVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.utility.AssertVertex;
import lombok.Getter;

import java.util.Arrays;
import java.util.List;

public interface BooleanVertex extends
    BooleanOperators<BooleanVertex>,
    TensorVertex<Boolean, BooleanTensor, BooleanVertex>,
    BaseBooleanTensor<BooleanVertex, IntegerVertex, DoubleVertex> {

    /////////////
    //// Vertex
    /////////////

    default void setValue(boolean value) {
        setValue(BooleanTensor.scalar(value));
    }

    default void setValue(boolean[] values) {
        setValue(BooleanTensor.create(values));
    }

    default void setAndCascade(boolean value) {
        setAndCascade(BooleanTensor.scalar(value));
    }

    default void setAndCascade(boolean[] values) {
        setAndCascade(BooleanTensor.create(values));
    }

    default void observe(boolean value) {
        observe(BooleanTensor.scalar(value));
    }

    default void observe(boolean[] values) {
        observe(BooleanTensor.create(values));
    }

    default boolean getValue(long... index) {
        return getValue().getValue(index);
    }

    /////////////
    //// Tensor Ops
    /////////////

    static BooleanVertex concat(int dimension, BooleanVertex... toConcat) {
        return new BooleanConcatenationVertex(dimension, toConcat);
    }

    default AssertVertex assertTrue() {
        return new AssertVertex(this);
    }

    default AssertVertex assertTrue(String errorMessage) {
        return new AssertVertex(this, errorMessage);
    }

    default BooleanVertex take(long... index) {
        return new BooleanTakeVertex(this, index);
    }

    @Override
    default List<BooleanVertex> split(int dimension, long... splitAtIndices) {
        return null;
    }

    class BooleanDiagVertex extends BooleanUnaryOpVertex<BooleanTensor> {

        @ExportVertexToPythonBindings
        public BooleanDiagVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor> inputVertex) {
            super(inputVertex);
        }

        @Override
        protected BooleanTensor op(BooleanTensor l) {
            return l.diag();
        }
    }

    @Override
    default BooleanVertex diag() {
        return new BooleanDiagVertex(this);
    }

    default BooleanVertex reshape(long... proposedShape) {
        return new BooleanReshapeVertex(this, proposedShape);
    }

    class BooleanPermuteVertex extends BooleanUnaryOpVertex<BooleanTensor> {
        private static final String REARRANGE = "arrange";

        @Getter(onMethod = @__({@SaveVertexParam(REARRANGE)}))
        private final int[] rearrange;

        @ExportVertexToPythonBindings
        public BooleanPermuteVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor> inputVertex,
                                    @LoadVertexParam(REARRANGE) int[] rearrange) {
            super(inputVertex);
            this.rearrange = rearrange;
        }

        @Override
        protected BooleanTensor op(BooleanTensor l) {
            return l.permute(rearrange);
        }
    }

    @Override
    default BooleanVertex permute(int... rearrange) {
        return new BooleanPermuteVertex(this, rearrange);
    }

    class BooleanBroadcastVertex extends BooleanUnaryOpVertex<BooleanTensor> {
        private static final String TO_SHAPE = "toShape";

        @Getter(onMethod = @__({@SaveVertexParam(TO_SHAPE)}))
        private final long[] toShape;

        @ExportVertexToPythonBindings
        public BooleanBroadcastVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor> inputVertex,
                                      @LoadVertexParam(TO_SHAPE) long[] toShape) {
            super(inputVertex);
            this.toShape = toShape;
        }

        @Override
        protected BooleanTensor op(BooleanTensor l) {
            return l.broadcast(toShape);
        }
    }

    @Override
    default BooleanVertex broadcast(long... toShape) {
        return new BooleanBroadcastVertex(this, toShape);
    }

    @Override
    default BooleanVertex notEqualTo(BooleanVertex that) {
        return new NotEqualsVertex<>(this, that);
    }

    @Override
    default BooleanVertex notEqualTo(Boolean value) {
        return notEqualTo(new ConstantBooleanVertex(value));
    }

    @Override
    default BooleanVertex elementwiseEquals(BooleanVertex that) {
        return new EqualsVertex<>(this, that);
    }

    @Override
    default BooleanVertex elementwiseEquals(Boolean value) {
        return elementwiseEquals(new ConstantBooleanVertex(value));
    }

    class BooleanGetBooleanIndexVertex extends BooleanUnaryOpVertex<BooleanTensor> {

        @ExportVertexToPythonBindings
        public BooleanGetBooleanIndexVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor> inputVertex) {
            super(inputVertex);
        }

        @Override
        protected BooleanTensor op(BooleanTensor l) {
            return l.get(l);
        }
    }

    @Override
    default BooleanVertex get(BooleanVertex booleanIndex) {
        return new BooleanGetBooleanIndexVertex(booleanIndex);
    }

    default BooleanVertex slice(int dimension, long index) {
        return new BooleanSliceVertex(this, dimension, index);
    }

    @Override
    default BooleanVertex slice(Slicer slicer) {
        return null;
    }

    /////////////
    //// Boolean Ops
    /////////////

    default BooleanVertex or(Vertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new OrBinaryVertex(this, those[0]);
        List<Vertex<BooleanTensor>> list = ImmutableList.<Vertex<BooleanTensor>>builder()
            .addAll(Arrays.asList(those))
            .add(this)
            .build();
        return new OrMultipleVertex(list);
    }

    @Override
    default BooleanVertex or(boolean that) {
        return this.or(new ConstantBooleanVertex(that));
    }

    @Override
    default BooleanVertex xor(BooleanVertex that) {
        return new XorBinaryVertex(this, that);
    }

    @Override
    default BooleanVertex or(BooleanVertex that) {
        return new OrBinaryVertex(this, that);
    }


    default BooleanVertex and(Vertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new AndBinaryVertex(this, those[0]);
        List<Vertex<BooleanTensor>> list = ImmutableList.<Vertex<BooleanTensor>>builder()
            .addAll(Arrays.asList(those))
            .add(this)
            .build();
        return new AndMultipleVertex(list);
    }

    @Override
    default BooleanVertex and(BooleanVertex that) {
        return new AndBinaryVertex(this, that);
    }

    @Override
    default BooleanVertex and(boolean that) {
        return this.and(new ConstantBooleanVertex(that));
    }

    @Override
    default BooleanVertex not() {
        return BooleanVertex.not(this);
    }

    class BooleanDoubleWhereVertex extends VertexImpl<DoubleTensor> implements DoubleVertex, NonProbabilistic<DoubleTensor> {
        private static final String INPUT_NAME = "inputName";
        private static final String TRUE_VALUE = "trueValue";
        private static final String FALSE_VALUE = "falseValue";

        @Getter(onMethod = @__({@SaveVertexParam(INPUT_NAME)}))
        private final BooleanVertex inputVertex;

        @Getter(onMethod = @__({@SaveVertexParam(TRUE_VALUE)}))
        private final DoubleVertex trueValue;

        @Getter(onMethod = @__({@SaveVertexParam(FALSE_VALUE)}))
        private final DoubleVertex falseValue;

        @ExportVertexToPythonBindings
        public BooleanDoubleWhereVertex(@LoadVertexParam(INPUT_NAME) BooleanVertex inputVertex,
                                        @LoadVertexParam(TRUE_VALUE) DoubleVertex trueValue,
                                        @LoadVertexParam(FALSE_VALUE) DoubleVertex falseValue) {
            this.inputVertex = inputVertex;
            this.trueValue = trueValue;
            this.falseValue = falseValue;
        }

        @Override
        public DoubleTensor calculate() {
            return inputVertex.getValue().doubleWhere(trueValue.getValue(), falseValue.getValue());
        }
    }

    @Override
    default DoubleVertex doubleWhere(DoubleVertex trueValue, DoubleVertex falseValue) {
        return new BooleanDoubleWhereVertex(this, trueValue, falseValue);
    }

    class BooleanIntegerWhereVertex extends VertexImpl<IntegerTensor> implements IntegerVertex, NonProbabilistic<IntegerTensor> {
        private static final String INPUT_NAME = "inputName";
        private static final String TRUE_VALUE = "trueValue";
        private static final String FALSE_VALUE = "falseValue";

        @Getter(onMethod = @__({@SaveVertexParam(INPUT_NAME)}))
        private final BooleanVertex inputVertex;

        @Getter(onMethod = @__({@SaveVertexParam(TRUE_VALUE)}))
        private final IntegerVertex trueValue;

        @Getter(onMethod = @__({@SaveVertexParam(FALSE_VALUE)}))
        private final IntegerVertex falseValue;

        @ExportVertexToPythonBindings
        public BooleanIntegerWhereVertex(@LoadVertexParam(INPUT_NAME) BooleanVertex inputVertex,
                                         @LoadVertexParam(TRUE_VALUE) IntegerVertex trueValue,
                                         @LoadVertexParam(FALSE_VALUE) IntegerVertex falseValue) {
            this.inputVertex = inputVertex;
            this.trueValue = trueValue;
            this.falseValue = falseValue;
        }

        @Override
        public IntegerTensor calculate() {
            return inputVertex.getValue().integerWhere(trueValue.getValue(), falseValue.getValue());
        }
    }

    @Override
    default IntegerVertex integerWhere(IntegerVertex trueValue, IntegerVertex falseValue) {
        return new BooleanIntegerWhereVertex(this, trueValue, falseValue);
    }

    class BooleanBooleanWhereVertex extends VertexImpl<BooleanTensor> implements BooleanVertex, NonProbabilistic<BooleanTensor> {
        private static final String INPUT_NAME = "inputName";
        private static final String TRUE_VALUE = "trueValue";
        private static final String FALSE_VALUE = "falseValue";

        @Getter(onMethod = @__({@SaveVertexParam(INPUT_NAME)}))
        private final BooleanVertex inputVertex;

        @Getter(onMethod = @__({@SaveVertexParam(TRUE_VALUE)}))
        private final BooleanVertex trueValue;

        @Getter(onMethod = @__({@SaveVertexParam(FALSE_VALUE)}))
        private final BooleanVertex falseValue;

        @ExportVertexToPythonBindings
        public BooleanBooleanWhereVertex(@LoadVertexParam(INPUT_NAME) BooleanVertex inputVertex,
                                         @LoadVertexParam(TRUE_VALUE) BooleanVertex trueValue,
                                         @LoadVertexParam(FALSE_VALUE) BooleanVertex falseValue) {
            this.inputVertex = inputVertex;
            this.trueValue = trueValue;
            this.falseValue = falseValue;
        }

        @Override
        public BooleanTensor calculate() {
            return inputVertex.getValue().booleanWhere(trueValue.getValue(), falseValue.getValue());
        }
    }

    @Override
    default BooleanVertex booleanWhere(BooleanVertex trueValue, BooleanVertex falseValue) {
        return new BooleanBooleanWhereVertex(this, trueValue, falseValue);
    }

    class AllTrueVertex extends BooleanUnaryOpVertex<BooleanTensor> {

        @ExportVertexToPythonBindings
        public AllTrueVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor> inputVertex) {
            super(inputVertex);
        }

        @Override
        protected BooleanTensor op(BooleanTensor l) {
            return l.allTrue();
        }
    }

    @Override
    default BooleanVertex allTrue() {
        return new AllTrueVertex(this);
    }

    class AllFalseVertex extends BooleanUnaryOpVertex<BooleanTensor> {

        @ExportVertexToPythonBindings
        public AllFalseVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor> inputVertex) {
            super(inputVertex);
        }

        @Override
        protected BooleanTensor op(BooleanTensor l) {
            return l.allFalse();
        }
    }

    @Override
    default BooleanVertex allFalse() {
        return new AllFalseVertex(this);
    }

    class AnyTrueVertex extends BooleanUnaryOpVertex<BooleanTensor> {

        @ExportVertexToPythonBindings
        public AnyTrueVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor> inputVertex) {
            super(inputVertex);
        }

        @Override
        protected BooleanTensor op(BooleanTensor l) {
            return l.anyTrue();
        }
    }

    @Override
    default BooleanVertex anyTrue() {
        return new AnyTrueVertex(this);
    }

    class AnyFalseVertex extends BooleanUnaryOpVertex<BooleanTensor> {

        @ExportVertexToPythonBindings
        public AnyFalseVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor> inputVertex) {
            super(inputVertex);
        }

        @Override
        protected BooleanTensor op(BooleanTensor l) {
            return l.anyFalse();
        }
    }

    @Override
    default BooleanVertex anyFalse() {
        return new AnyFalseVertex(this);
    }

    class BooleanToDoubleMaskVertex extends VertexImpl<DoubleTensor> implements DoubleVertex, NonProbabilistic<DoubleTensor> {
        private static final String INPUT_NAME = "inputName";

        @Getter(onMethod = @__({@SaveVertexParam(INPUT_NAME)}))
        private final BooleanVertex inputVertex;

        @ExportVertexToPythonBindings
        public BooleanToDoubleMaskVertex(@LoadVertexParam(INPUT_NAME) BooleanVertex inputVertex) {
            this.inputVertex = inputVertex;
        }

        @Override
        public DoubleTensor calculate() {
            return inputVertex.getValue().toDoubleMask();
        }
    }

    @Override
    default DoubleVertex toDoubleMask() {
        return new BooleanToDoubleMaskVertex(this);
    }

    class BooleanToIntegerMaskVertex extends VertexImpl<IntegerTensor> implements IntegerVertex, NonProbabilistic<IntegerTensor> {
        private static final String INPUT_NAME = "inputName";

        @Getter(onMethod = @__({@SaveVertexParam(INPUT_NAME)}))
        private final BooleanVertex inputVertex;

        @ExportVertexToPythonBindings
        public BooleanToIntegerMaskVertex(@LoadVertexParam(INPUT_NAME) BooleanVertex inputVertex) {
            this.inputVertex = inputVertex;
        }

        @Override
        public IntegerTensor calculate() {
            return inputVertex.getValue().toIntegerMask();
        }
    }

    @Override
    default IntegerVertex toIntegerMask() {
        return new BooleanToIntegerMaskVertex(this);
    }

    static BooleanVertex not(Vertex<BooleanTensor> vertex) {
        return new NotBinaryVertex(vertex);
    }

}
