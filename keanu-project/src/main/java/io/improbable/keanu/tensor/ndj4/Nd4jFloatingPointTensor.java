package io.improbable.keanu.tensor.ndj4;

import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.TensorMulByMatrixMul;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.LogX;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.PowPairwise;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ACosh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ASinh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Expm1;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Log1p;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Tan;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;

public abstract class Nd4jFloatingPointTensor<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>> extends Nd4jNumberTensor<T, TENSOR> implements FloatingPointTensor<T, TENSOR> {

    public Nd4jFloatingPointTensor(INDArray tensor) {
        super(tensor);
    }

    @Override
    public TENSOR matrixInverse() {
        return create(InvertMatrix.invert(tensor, false));
    }

    @Override
    public TENSOR choleskyDecomposition() {
        INDArray dup = tensor.dup();
        Nd4j.getBlasWrapper().lapack().potrf(dup, true);
        return create(dup);
    }

    @Override
    public TENSOR matrixMultiply(TENSOR value) {
        TensorShapeValidation.getMatrixMultiplicationResultingShape(tensor.shape(), value.getShape());
        INDArray mmulResult = tensor.mmul(getTensor(value));
        return create(mmulResult);
    }

    @Override
    public TENSOR tensorMultiply(TENSOR value, int[] dimsLeft, int[] dimsRight) {
        return TensorMulByMatrixMul.tensorMmul(getThis(), value, dimsLeft, dimsRight);
    }

    @Override
    public TENSOR sqrtInPlace() {
        Transforms.sqrt(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR logInPlace() {
        Transforms.log(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR reciprocalInPlace() {
        tensor.rdivi(1.0);
        return getThis();
    }

    @Override
    public TENSOR sinInPlace() {
        Transforms.sin(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR cosInPlace() {
        Transforms.cos(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR tanInPlace() {
        Nd4j.getExecutioner().exec(new Tan(tensor, tensor));
        return getThis();
    }

    @Override
    public TENSOR atanInPlace() {
        Transforms.atan(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR atan2InPlace(T y) {
        tensor = Transforms.atan2(tensor, Nd4j.scalar(y).broadcast(this.tensor.shape()));
        return getThis();
    }

    @Override
    public TENSOR atan2InPlace(TENSOR y) {
        if (y.isScalar()) {
            tensor = Transforms.atan2(tensor, getTensor(y).broadcast(this.tensor.shape()));
        } else {
            tensor = INDArrayShim.atan2(tensor, getTensor(y));
        }
        return getThis();
    }

    @Override
    public TENSOR asinInPlace() {
        Transforms.asin(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR acosInPlace() {
        Transforms.acos(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR sinhInPlace() {
        Transforms.sinh(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR coshInPlace() {
        Transforms.cosh(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR tanhInPlace() {
        Transforms.tanh(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR asinhInPlace() {
        Nd4j.getExecutioner().execAndReturn(new ASinh(tensor));
        return getThis();
    }

    @Override
    public TENSOR acoshInPlace() {
        Nd4j.getExecutioner().execAndReturn(new ACosh(tensor));
        return getThis();
    }

    @Override
    public TENSOR atanhInPlace() {
        Transforms.atanh(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR expInPlace() {
        Transforms.exp(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR log1pInPlace() {
        Nd4j.getExecutioner().exec(new Log1p(tensor));
        return getThis();
    }

    @Override
    public TENSOR log2InPlace() {
        Nd4j.getExecutioner().exec(new LogX(tensor, 2));
        return getThis();
    }

    @Override
    public TENSOR log10InPlace() {
        Nd4j.getExecutioner().exec(new LogX(tensor, 10));
        return getThis();
    }

    @Override
    public TENSOR exp2InPlace() {
        INDArray indArray = Nd4j.valueArrayOf(tensor.shape(), 2.0);
        Nd4j.getExecutioner().exec(new PowPairwise(indArray, tensor, tensor));
        return getThis();
    }

    @Override
    public TENSOR expM1InPlace() {
        Nd4j.getExecutioner().exec(new Expm1(tensor));
        return getThis();
    }

    @Override
    public TENSOR standardizeInPlace() {
        tensor.subi(average()).divi(standardDeviation());
        return getThis();
    }

    @Override
    public TENSOR setAllInPlace(T value) {
        this.tensor.assign(value);
        return getThis();
    }

    @Override
    public TENSOR clampInPlace(TENSOR min, TENSOR max) {
        return minInPlace(max).maxInPlace(min);
    }

    public TENSOR ceilInPlace() {
        Transforms.ceil(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR floorInPlace() {
        Transforms.floor(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR roundInPlace() {
        Transforms.round(tensor, false);
        return getThis();
    }

    @Override
    public TENSOR sigmoidInPlace() {
        Transforms.sigmoid(tensor, false);
        return getThis();
    }

    @Override
    public T determinant() {
        INDArray dup = tensor.dup();
        double[][] asMatrix = dup.toDoubleMatrix();
        RealMatrix matrix = new Array2DRowRealMatrix(asMatrix);
        return getNumber(new LUDecomposition(matrix).getDeterminant());
    }

    @Override
    public T product() {
        return getNumber(tensor.prod().getNumber(0));
    }

    @Override
    public BooleanTensor notNaN() {
        return this.elementwiseEquals(this);
    }

}
