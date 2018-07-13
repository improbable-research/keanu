package io.improbable.keanu.vertices;

public class NotObservable<T> implements Observable<T> {
    // package private
    NotObservable() {}

    @Override
    public void observe(T value) {
        throw new UnsupportedOperationException("This type of vertex does not support being observed");
    }

    @Override
    public void unobserve() {
    }

    @Override
    public boolean isObserved() {
        return false;
    }
}
