package io.improbable.keanu.kotlin

import io.improbable.keanu.DeterministicRule
import io.improbable.keanu.vertices.dbl.DoubleVertex
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex
import io.improbable.keanu.vertices.intgr.IntegerVertex
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex
import junit.framework.TestCase.assertEquals
import org.junit.Rule
import org.junit.Test


class OperatorOverloadingTest {

    @Rule
    @JvmField
    var deterministicRule = DeterministicRule()

    @Test
    fun doubleVertexPlus() {
        val a = GaussianVertex(0.0, 1.0)
        val b = GaussianVertex(0.0, 1.0)

        val e1 = a.value + b.value
        val r1 = a + b
        assertEquals(e1, r1.value)

        val e2 = a.value + 2.0
        val r2 = a + 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 + a.value
        val r3 = 2.0 + a
        assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticDoublePlus() {
        val a = ArithmeticDouble(1.0)
        val b = ArithmeticDouble(1.5)

        val e1 = a.value + b.value
        val r1 = a + b
        assertEquals(e1, r1.value)

        val e2 = a.value + 2.0
        val r2 = a + 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 + a.value
        val r3 = 2.0 + a
        assertEquals(e3, r3.value)
    }

    @Test
    fun integerVertexPlus() {
        val a = PoissonVertex(1.0)
        val b = PoissonVertex(2.0)

        val e1 = a.value + b.value
        val r1 = a + b
        assertEquals(e1, r1.value)

        val e2 = a.value + 2
        val r2 = a + 2
        assertEquals(e2, r2.value)

        val e3 = 2 + a.value
        val r3 = 2 + a
        assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticIntegerPlus() {
        val a = ArithmeticInteger(1)
        val b = ArithmeticInteger(2)

        val e1 = a.value + b.value
        val r1 = a + b
        assertEquals(e1, r1.value)

        val e2 = a.value + 2
        val r2 = a + 2
        assertEquals(e2, r2.value)

        val e3 = 2 + a.value
        val r3 = 2 + a
        assertEquals(e3, r3.value)
    }

    @Test
    fun doubleVertexMinus() {
        val a = GaussianVertex(0.0, 1.0)
        val b = GaussianVertex(0.0, 1.0)

        val e1 = a.value - b.value
        val r1 = a - b
        assertEquals(e1, r1.value)

        val e2 = a.value - 2.0
        val r2 = a - 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 - a.value
        val r3 = 2.0 - a
        assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticDoubleMinus() {
        val a = ArithmeticDouble(1.0)
        val b = ArithmeticDouble(2.0)

        val e1 = a.value - b.value
        val r1 = a - b
        assertEquals(e1, r1.value)

        val e2 = a.value - 2.0
        val r2 = a - 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 - a.value
        val r3 = 2.0 - a
        assertEquals(e3, r3.value)
    }

    @Test
    fun integerVertexMinus() {
        val a = PoissonVertex(1.0)
        val b = PoissonVertex(2.0)

        val e1 = a.value - b.value
        val r1 = a - b
        assertEquals(e1, r1.value)

        val e2 = a.value - 2
        val r2 = a - 2
        assertEquals(e2, r2.value)

        val e3 = 2 - a.value
        val r3 = 2 - a
        assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticIntegerMinus() {
        val a = ArithmeticInteger(1)
        val b = ArithmeticInteger(2)

        val e1 = a.value - b.value
        val r1 = a - b
        assertEquals(e1, r1.value)

        val e2 = a.value - 2
        val r2 = a - 2
        assertEquals(e2, r2.value)

        val e3 = 2 - a.value
        val r3 = 2 - a
        assertEquals(e3, r3.value)
    }

    @Test
    fun doubleVertexUnaryMinus() {
        val a = GaussianVertex(0.0, 1.0)

        val e1 = -a.value
        val r1 = -a
        assertEquals(e1, r1.value)
    }

    @Test
    fun arithmeticDoubleUnaryMinus() {
        val a = ArithmeticDouble(1.0)

        val e1 = -a.value
        val r1 = -a
        assertEquals(e1, r1.value)
    }

    @Test
    fun integerVertexUnaryMinus() {
        val a = PoissonVertex(1.0)

        val e1 = -a.value
        val r1 = -a
        assertEquals(e1, r1.value)
    }

    @Test
    fun arithmeticIntegerUnaryMinus() {
        val a = ArithmeticInteger(2)

        val e1 = -a.value
        val r1 = -a
        assertEquals(e1, r1.value)
    }

    @Test
    fun doubleVertexTimes() {
        val a = GaussianVertex(0.0, 1.0)
        val b = GaussianVertex(0.0, 1.0)

        val e1 = a.value * b.value
        val r1 = a * b
        assertEquals(e1, r1.value)

        val e2 = a.value * 2.0
        val r2 = a * 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 * a.value
        val r3 = 2.0 * a
        assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticDoubleTimes() {
        val a = ArithmeticDouble(1.0)
        val b = ArithmeticDouble(2.0)

        val e1 = a.value * b.value
        val r1 = a * b
        assertEquals(e1, r1.value)

        val e2 = a.value * 2.0
        val r2 = a * 2.0
        assertEquals(e2, r2.value)
    }

    @Test
    fun integerVertexTimes() {
        val a = PoissonVertex(1.0)
        val b = PoissonVertex(2.0)

        val e1 = a.value * b.value
        val r1 = a * b
        assertEquals(e1, r1.value)

        val e2 = a.value * 2
        val r2 = a * 2
        assertEquals(e2, r2.value)

        val e3 = 2 * a.value
        val r3 = 2 * a
        assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticIntegerTimes() {
        val a = ArithmeticInteger(1)
        val b = ArithmeticInteger(2)

        val e1 = a.value * b.value
        val r1 = a * b
        assertEquals(e1, r1.value)

        val e2 = a.value * 2
        val r2 = a * 2
        assertEquals(e2, r2.value)

        val e3 = 2 * a.value
        val r3 = 2 * a
        assertEquals(e3, r3.value)
    }

    @Test
    fun doubleVertexDivide() {
        val a = GaussianVertex(0.0, 1.0)
        val b = ConstantDoubleVertex(2.0)

        val e1 = a.value / b.value
        val r1 = a / b
        assertEquals(e1, r1.value)

        val e2 = a.value / 2.0
        val r2 = a / 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 / a.value
        val r3 = 2.0 / a
        assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticDoubleDivide() {
        val a = ArithmeticDouble(1.0)
        val b = ArithmeticDouble(2.0)

        val e1 = a.value / b.value
        val r1 = a / b
        assertEquals(e1, r1.value)

        val e2 = a.value / 2.0
        val r2 = a / 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 / a.value
        val r3 = 2.0 / a
        assertEquals(e3, r3.value)
    }


    @Test
    fun integerVertexDivide() {
        val a = PoissonVertex(1.0)
        val b = ConstantIntegerVertex(2)

        val e1 = a.value / b.value
        val r1 = a / b
        assertEquals(e1, r1.value)

        val e2 = a.value / 2
        val r2 = a / 2
        assertEquals(e2, r2.value)

        val e3 = 2 / (a.value + 1)
        val r3 = 2 / (a + 1)
        assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticIntegerDivide() {
        val a = ArithmeticInteger(1)
        val b = ArithmeticInteger(2)

        val e1 = a.value / b.value
        val r1 = a / b
        assertEquals(e1, r1.value)

        val e2 = a.value / 2
        val r2 = a / 2
        assertEquals(e2, r2.value)

        val e3 = 2 / a.value
        val r3 = 2 / a
        assertEquals(e3, r3.value)
    }

    @Test
    fun doubleVertexNestedOperators() {
        val a = 4.0
        val b = GaussianVertex(0.0, 1.0)
        val c = GaussianVertex(0.0, 1.0)
        val d = ConstantDoubleVertex(1.0)
        val e = UniformVertex(1.0, 10.0)
        val f = 10.0
        val g = 5.0

        val result = a / ((((b + c - d) / e) * f) - g)
        val expected = a / ((((b.value + c.value - d.value) / e.value) * f) - g)
        assertEquals(expected, result.value)
    }

    @Test
    fun arithmeticDoubleNestedOperators() {
        val a = 4.0
        val b = ArithmeticDouble(1.0)
        val c = ArithmeticDouble(2.0)
        val d = ArithmeticDouble(3.0)
        val e = ArithmeticDouble(10.0)
        val f = 10.0
        val g = 5.0

        val result = a / ((((b + c - d) / e) * f) - g)
        val expected = a / ((((b.value + c.value - d.value) / e.value) * f) - g)
        assertEquals(expected, result.value)
    }

    @Test
    fun integerVertexNestedOperators() {
        val a = 4
        val b = PoissonVertex(1.0)
        val c = PoissonVertex(2.0)
        val d = ConstantIntegerVertex(1)
        val e = UniformIntVertex(1, 10)
        val f = 10
        val g = 5

        val result = a / ((((b + c - d) / e) * f) - g)
        val expected = a / ((((b.value + c.value - d.value) / e.value) * f) - g)
        assertEquals(expected, result.value)
    }

    @Test
    fun arithmeticIntegerNestedOperators() {
        val a = 4
        val b = ArithmeticInteger(1)
        val c = ArithmeticInteger(2)
        val d = ArithmeticInteger(3)
        val e = ArithmeticInteger(10)
        val f = 10
        val g = 5

        val result = a / ((((b + c - d) / e) * f) - g)
        val expected = a / ((((b.value + c.value - d.value) / e.value) * f) - g)
        assertEquals(expected, result.value)
    }

    @Test
    fun runSimpleModel() {
        val modelA = SimpleModel<ArithmeticDouble, ArithmeticInteger>()

        val ad1 = ArithmeticDouble(1.0)
        val ad2 = ArithmeticDouble(1.5)
        val resultD1 = modelA.add(ad1, ad2)

        val ai1 = ArithmeticInteger(1)
        val ai2 = ArithmeticInteger(2)
        val resultI1 = modelA.add(ai1, ai2)

        val modelB = SimpleModel<DoubleVertex, IntegerVertex>()

        val dv1 = ConstantDoubleVertex(1.0)
        val dv2 = ConstantDoubleVertex(1.5)
        val resultD2 = modelB.add(dv1, dv2)

        val iv1 = ConstantIntegerVertex(1)
        val iv2 = ConstantIntegerVertex(2)
        val resultI2 = modelB.add(iv1, iv2)

        assertEquals(resultD1.value, resultD2.value.scalar())
        assertEquals(resultI1.value, resultI2.value.scalar())
    }

    class SimpleModel<A : DoubleOperators<A>, B : IntegerOperators<B>> {

        fun add(a: A, b: A): A {
            return a + b
        }

        fun add(a: B, b: B): B {
            return a + b
        }
    }

    @Test
    fun vertexOperatorTest() {
        val a = ConstantDoubleVertex(0.123)

        assertEquals(Math.acos(a.value.scalar()), acos(a).value.scalar())
        assertEquals(Math.asin(a.value.scalar()), asin(a).value.scalar())
        assertEquals(Math.cos(a.value.scalar()), cos(a).value.scalar())
        assertEquals(Math.sin(a.value.scalar()), sin(a).value.scalar())
        assertEquals(Math.exp(a.value.scalar()), exp(a).value.scalar())
        assertEquals(Math.log(a.value.scalar()), log(a).value.scalar())
        assertEquals(Math.pow(a.value.scalar(), 2.345), pow(a, 2.345).value.scalar())
        assertEquals(Math.pow(a.value.scalar(), a.value.scalar()), pow(a, a).value.scalar())
    }

    @Test
    fun arithmeticDoubleOperatorTest() {
        val a = ArithmeticDouble(0.123)

        assertEquals(Math.acos(a.value), acos(a).value)
        assertEquals(Math.asin(a.value), asin(a).value)
        assertEquals(Math.cos(a.value), cos(a).value)
        assertEquals(Math.sin(a.value), sin(a).value)
        assertEquals(Math.exp(a.value), exp(a).value)
        assertEquals(Math.log(a.value), log(a).value)
        assertEquals(Math.pow(a.value, 2.345), pow(a, 2.345).value)
        assertEquals(Math.pow(a.value, a.value), pow(a, a).value)
    }


}
