package io.improbable.keanu.kotlin

import io.improbable.keanu.DeterministicRule
import io.improbable.keanu.vertices.VertexMatchers.hasTensorValue
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex
import junit.framework.TestCase
import org.hamcrest.MatcherAssert.assertThat
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class BooleanOperatorOverloadingTest {

    @Rule
    @JvmField
    var deterministicRule = DeterministicRule()

    private val arithmeticFalse = ArithmeticBoolean(false)
    private val arithmeticTrue = ArithmeticBoolean(true)

    @Test
    fun booleanVertexAnd() {
        val vertex1 = BernoulliVertex(0.5)
        val vertex2 = BernoulliVertex(0.5)

        assertThat(vertex1 and vertex2, hasTensorValue(vertex1.value and vertex2.value))
        assertThat(vertex1 and false, hasTensorValue(vertex1.value and false))
        assertThat(true and vertex1, hasTensorValue(vertex1.value and true))
    }

    @Test
    fun arithmeticBooleanAnd() {
        assertEquals(arithmeticFalse and arithmeticTrue, arithmeticFalse)
        assertEquals(arithmeticFalse and true, arithmeticFalse)
        assertEquals(true and arithmeticTrue, arithmeticTrue)
    }

    @Test
    fun booleanVertexOr() {
        val vertex1 = BernoulliVertex(0.5)
        val vertex2 = BernoulliVertex(0.5)

        assertThat(vertex1 or vertex2, hasTensorValue(vertex1.value or vertex2.value))
        assertThat(vertex1 or false, hasTensorValue(vertex1.value or false))
        assertThat(true or vertex1, hasTensorValue(vertex1.value or true))
    }

    @Test
    fun arithmeticBooleanOr() {
        assertEquals(arithmeticFalse or arithmeticTrue, arithmeticTrue)
        assertEquals(arithmeticFalse or false, arithmeticFalse)
        assertEquals(false or arithmeticTrue, arithmeticTrue)
    }

    @Test
    fun boolVertexPrefixOperators() {
        val boolVertex = ConstantBoolVertex(true)

        TestCase.assertEquals(false, !boolVertex.value.scalar())
    }

    @Test
    fun arithmeticBooleanPrefixOperators() {
        val arithmeticBoolean = ArithmeticBoolean(true)

        TestCase.assertEquals(false, !arithmeticBoolean.value)
    }

}
