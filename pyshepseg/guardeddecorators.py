"""
A dreadful hack to get around the fact that the numpydoc Sphinx extension
does not play well with numba's jitclass decorator.

If we are running with Sphinx, then fake the jitclass decorator so that it
just returns the class it is decorating.

"""
import sys

if 'sphinx' not in sys.modules:
    from numba.experimental import jitclass
else:
    def jitclass(cls_or_spec=None, spec=None):
        """
        Our fake jitclass decorator. Hacked from the real one
        in numba.

        Returns
        -------
        If used as a decorator, returns a callable that takes a class
        object and returns the same class. In short, this decorator does
        nothing at all.

        """

        if (cls_or_spec is not None and
            spec is None and
                not isinstance(cls_or_spec, type)):
            # Used like
            # @jitclass([("x", intp)])
            # class Foo:
            #     ...
            spec = cls_or_spec
            cls_or_spec = None

        def wrap(cls):
            return cls

        if cls_or_spec is None:
            return wrap
        else:
            return wrap(cls_or_spec)
