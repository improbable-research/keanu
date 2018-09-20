package io.improbable.keanu.vertices;

/**
 * A Proxy vertex is one with a single parent. It is expected that all vertex operations will be delegated to the
 * parent.  It typically has no parents at creation time - the parent is node is "hooked up" at a later point.
 */
public interface ProxyVertex<T extends Vertex<?>> {
    void setParent(T newParent);

    boolean hasParent();
}
