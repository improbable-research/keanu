package io.improbable.keanu.vertices.dbltensor;

public class SimpleDoubleTensor implements DoubleTensor {

    private Double value;
    private int[] shape;

    public SimpleDoubleTensor(double value) {
        this.value = value;
        this.shape = SCALAR_SHAPE;
    }

    public SimpleDoubleTensor(int[] shape) {
        this.value = null;
        this.shape = shape;
    }

    @Override
    public DoubleTensor duplicate() {
        return new SimpleDoubleTensor(value);
    }

    @Override
    public double getValue(int[] index) {
        return value;
    }

    @Override
    public void setValue(double value, int[] index) {
        this.value = value;
    }

    @Override
    public double sum() {
        return value;
    }

    @Override
    public double scalar() {
        return value;
    }

    @Override
    public DoubleTensor reciprocal() {
        return this.duplicate().reciprocalInPlace();
    }

    @Override
    public DoubleTensor minus(double that) {
        return this.duplicate().minusInPlace(that);
    }

    @Override
    public DoubleTensor plus(double that) {
        return this.duplicate().plusInPlace(that);
    }

    @Override
    public DoubleTensor times(double that) {
        return this.duplicate().timesInPlace(that);
    }

    @Override
    public DoubleTensor div(double that) {
        return this.duplicate().divInPlace(that);
    }

    @Override
    public DoubleTensor pow(DoubleTensor exponent) {
        return this.duplicate().powInPlace(exponent);
    }

    @Override
    public DoubleTensor pow(double exponent) {
        return this.duplicate().powInPlace(exponent);
    }

    @Override
    public DoubleTensor log() {
        return this.duplicate().logInPlace();
    }

    @Override
    public DoubleTensor sin() {
        return this.duplicate().sinInPlace();
    }

    @Override
    public DoubleTensor cos() {
        return this.duplicate().cosInPlace();
    }

    @Override
    public DoubleTensor asin() {
        return this.duplicate().asin();
    }

    @Override
    public DoubleTensor acos() {
        return this.duplicate().acos();
    }

    @Override
    public DoubleTensor exp() {
        return this.duplicate().exp();
    }

    @Override
    public DoubleTensor minus(DoubleTensor that) {
        return this.duplicate().minusInPlace(that);
    }

    @Override
    public DoubleTensor plus(DoubleTensor that) {
        return this.duplicate().plusInPlace(that);
    }

    @Override
    public DoubleTensor times(DoubleTensor that) {
        return this.duplicate().timesInPlace(that);
    }

    @Override
    public DoubleTensor div(DoubleTensor that) {
        return this.duplicate().divInPlace(that);
    }

    @Override
    public DoubleTensor unaryMinus() {
        return this.duplicate().unaryMinusInPlace();
    }

    @Override
    public DoubleTensor getGreaterThanMask(DoubleTensor greaterThanThis) {
        if (greaterThanThis.isScalar()) {
            return new SimpleDoubleTensor(value > greaterThanThis.scalar() ? 1 : 0);
        }
        throw new IllegalArgumentException("Only scalar tensors supported");
    }

    @Override
    public DoubleTensor getLessThanOrEqualToMask(DoubleTensor lessThanThis) {
        if (lessThanThis.isScalar()) {
            return new SimpleDoubleTensor(value <= lessThanThis.scalar() ? 1 : 0);
        }
        throw new IllegalArgumentException("Only scalar tensors supported");
    }

    @Override
    public DoubleTensor setWithMaskInPlace(DoubleTensor withMask, double valueToApply) {
        if (withMask.isScalar()) {
            this.value = withMask.scalar() == 1.0 ? valueToApply : this.value;
        } else {
            throw new IllegalArgumentException("Only scalar tensors supported");
        }
        return this;
    }

    @Override
    public DoubleTensor setWithMask(DoubleTensor mask, double value) {
        return this.duplicate().setWithMaskInPlace(mask, value);
    }

    @Override
    public DoubleTensor reciprocalInPlace() {
        value = 1.0 / value;
        return this;
    }

    @Override
    public DoubleTensor minusInPlace(double that) {
        value = value - that;
        return this;
    }

    @Override
    public DoubleTensor plusInPlace(double that) {
        value = value + that;
        return this;
    }

    @Override
    public DoubleTensor timesInPlace(double that) {
        value = value * that;
        return this;
    }

    @Override
    public DoubleTensor divInPlace(double that) {
        value = value / that;
        return this;
    }

    @Override
    public DoubleTensor powInPlace(DoubleTensor exponent) {
        if (exponent.isScalar()) {
            value = Math.pow(value, exponent.scalar());
        } else {
            throw new IllegalArgumentException("Only scalar tensors supported");
        }

        return this;
    }

    @Override
    public DoubleTensor powInPlace(double exponent) {
        value = Math.pow(value, exponent);
        return this;
    }

    @Override
    public DoubleTensor logInPlace() {
        value = Math.log(value);
        return this;
    }

    @Override
    public DoubleTensor sinInPlace() {
        value = Math.sin(value);
        return this;
    }

    @Override
    public DoubleTensor cosInPlace() {
        value = Math.cos(value);
        return this;
    }

    @Override
    public DoubleTensor asinInPlace() {
        value = Math.asin(value);
        return this;
    }

    @Override
    public DoubleTensor acosInPlace() {
        value = Math.acos(value);
        return this;
    }

    @Override
    public DoubleTensor expInPlace() {
        value = Math.exp(value);
        return this;
    }

    @Override
    public DoubleTensor minusInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            minusInPlace(that.scalar());
        } else {
            throw new IllegalArgumentException("Only scalar tensors supported");
        }
        return this;
    }

    @Override
    public DoubleTensor plusInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            plusInPlace(that.scalar());
        } else {
            throw new IllegalArgumentException("Only scalar tensors supported");
        }
        return this;
    }

    @Override
    public DoubleTensor timesInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            timesInPlace(that.scalar());
        } else {
            throw new IllegalArgumentException("Only scalar tensors supported");
        }
        return this;
    }

    @Override
    public DoubleTensor divInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            divInPlace(that.scalar());
        } else {
            throw new IllegalArgumentException("Only scalar tensors supported");
        }
        return this;
    }

    @Override
    public DoubleTensor unaryMinusInPlace() {
        value = -value;
        return this;
    }

    @Override
    public FlattenedView getFlattenedView() {
        return new SimpleFlattenedView(value);
    }

    @Override
    public int getRank() {
        return shape.length;
    }

    @Override
    public int[] getShape() {
        return shape;
    }

    @Override
    public int getLength() {
        if (shape.length == 0) {
            return 0;
        } else {
            int prod = 1;
            for (int dim : shape) {
                prod *= dim;
            }
            return prod;
        }
    }

    @Override
    public boolean isShapePlaceholder() {
        return value == null;
    }

    private static class SimpleFlattenedView implements FlattenedView {

        private double value;

        public SimpleFlattenedView(double value) {
            this.value = value;
        }

        @Override
        public long size() {
            return 1;
        }

        @Override
        public double get(long index) {
            if (index != 0) {
                throw new IndexOutOfBoundsException();
            }
            return value;
        }

        @Override
        public void set(long index, double value) {
            if (index != 0) {
                throw new IndexOutOfBoundsException();
            }
            this.value = value;
        }

        @Override
        public double[] asArray() {
            return new double[]{value};
        }
    }
}
