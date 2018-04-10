package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;

public class If {

    public static IfThenBuilder isTrue(Vertex<Boolean> predicate) {
        return new IfThenBuilder(predicate);
    }

    public static class IfThenBuilder {
        private final Vertex<Boolean> predicate;

        public IfThenBuilder(Vertex<Boolean> predicate) {
            this.predicate = predicate;
        }

        public <T> IfThenElseBuilder<T> then(Vertex<T> thn) {
            return new IfThenElseBuilder<>(predicate, thn);
        }
    }

    public static class IfThenElseBuilder<T> {

        private final Vertex<Boolean> predicate;
        private final Vertex<T> thn;

        public IfThenElseBuilder(Vertex<Boolean> predicate, Vertex<T> thn) {
            this.predicate = predicate;
            this.thn = thn;
        }

        public IfVertex<T> orElse(Vertex<T> els) {
            return new IfVertex<>(predicate, thn, els);
        }
    }
}
