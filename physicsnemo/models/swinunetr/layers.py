import torch 
from collections.abc import Callable
from typing import Any
from utils import look_up_option
from component_store import ComponentStore
from physicsnemo.models.diffusion.layers import Conv2d
from collections.abc import Sequence
import torch.nn as nn
import inspect
import numpy as np
from utils import issequenceiterable


class LayerFactory(ComponentStore):
    """
    Factory object for creating layers, this uses given factory functions to actually produce the types or constructing
    callables. These functions are referred to by name and can be added at any time.
    """

    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self.__doc__ = (
            f"Layer Factory '{name}': {description}\n".strip()
            + "\nPlease see :py:class:`monai.networks.layers.split_args` for additional args parsing."
            + "\n\nThe supported members are:"
        )

    def add_factory_callable(self, name: str, func: Callable, desc: str | None = None) -> None:
        """
        Add the factory function to this object under the given name, with optional description.
        """
        description: str = desc or func.__doc__ or ""
        self.add(name.upper(), description, func)
        # append name to the docstring
        assert self.__doc__ is not None
        self.__doc__ += f"{', ' if len(self.names)>1 else ' '}``{name}``"

    def add_factory_class(self, name: str, cls: type, desc: str | None = None) -> None:
        """
        Adds a factory function which returns the supplied class under the given name, with optional description.
        """
        self.add_factory_callable(name, lambda x=None: cls, desc)

    def factory_function(self, name: str) -> Callable:
        """
        Decorator for adding a factory function with the given name.
        """

        def _add(func: Callable) -> Callable:
            self.add_factory_callable(name, func)
            return func

        return _add

    def get_constructor(self, factory_name: str, *args) -> Any:
        """
        Get the constructor for the given factory name and arguments.

        Raises:
            TypeError: When ``factory_name`` is not a ``str``.

        """

        if not isinstance(factory_name, str):
            raise TypeError(f"factory_name must a str but is {type(factory_name).__name__}.")

        component = look_up_option(factory_name.upper(), self.components)

        return component.value(*args)

    def __getitem__(self, args) -> Any:
        """
        Get the given name or name/arguments pair. If `args` is a callable it is assumed to be the constructor
        itself and is returned, otherwise it should be the factory name or a pair containing the name and arguments.
        """

        # `args[0]` is actually a type or constructor
        if callable(args):
            return args

        # `args` is a factory name or a name with arguments
        if isinstance(args, str):
            name_obj, args = args, ()
        else:
            name_obj, *args = args

        return self.get_constructor(name_obj, *args)

    def __getattr__(self, key):
        """
        If `key` is a factory name, return it, otherwise behave as inherited. This allows referring to factory names
        as if they were constants, eg. `Fact.FOO` for a factory Fact with factory function foo.
        """

        if key in self.components:
            return key

        return super().__getattribute__(key)


def split_args(args):
    """
    Split arguments in a way to be suitable for using with the factory types. If `args` is a string it's interpreted as
    the type name.

    Args:
        args (str or a tuple of object name and kwarg dict): input arguments to be parsed.

    Raises:
        TypeError: When ``args`` type is not in ``Union[str, Tuple[Union[str, Callable], dict]]``.

    Examples::

        >>> act_type, args = split_args("PRELU")
        >>> monai.networks.layers.Act[act_type]
        <class 'torch.nn.modules.activation.PReLU'>

        >>> act_type, args = split_args(("PRELU", {"num_parameters": 1, "init": 0.25}))
        >>> monai.networks.layers.Act[act_type](**args)
        PReLU(num_parameters=1)

    """

    if isinstance(args, str):
        return args, {}
    name_obj, name_args = args

    if not (isinstance(name_obj, str) or callable(name_obj)) or not isinstance(name_args, dict):
        msg = "Layer specifiers must be single strings or pairs of the form (name/object-types, argument dict)"
        raise TypeError(msg)

    return name_obj, name_args



Dropout = LayerFactory(name="Dropout layers", description="Factory for creating dropout layers.")
Norm = LayerFactory(name="Normalization layers", description="Factory for creating normalization layers.")
Conv = LayerFactory(name="Convolution layers", description="Factory for creating convolution layers.")
Pool = LayerFactory(name="Pooling layers", description="Factory for creating pooling layers.")
Pad = LayerFactory(name="Padding layers", description="Factory for creating padding layers.")
RelPosEmbedding = LayerFactory(
    name="Relative positional embedding layers",
    description="Factory for creating relative positional embedding factory",
)


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    stride: Sequence[int] | int = 1,
    act: tuple | str | None = None,
    norm: tuple | str | None = None,
    dropout: tuple | str | float | None = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
) -> nn.Module:
    """
    Create a convolution layer using PhysicsNEMO's Conv2d.
    Specific for 2D inputs with stride=1.
    """
    # Get kernel size (PhysicsNEMO uses single int)
    if isinstance(kernel_size, (list, tuple)):
        kernel = kernel_size[0]
    else:
        kernel = int(kernel_size)
    
    # Use PhysicsNEMO Conv2d
    if is_transposed:
        # Transposed conv (upsampling)
        return Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel,
            bias=bias,
            up=True,
            down=False,
        )
    else:
        # Regular conv (stride=1)
        return Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel,
            bias=bias,
            up=False,
            down=False,
        )
        
        
def get_norm_layer(name: tuple | str, spatial_dims: int | None = 1, channels: int | None = 1):
    """
    Create a normalization layer instance.

    For example, to create normalization layers:

    .. code-block:: python

        from monai.networks.layers import get_norm_layer

        g_layer = get_norm_layer(name=("group", {"num_groups": 1}))
        n_layer = get_norm_layer(name="instance", spatial_dims=2)

    Args:
        name: a normalization type string or a tuple of type string and parameters.
        spatial_dims: number of spatial dimensions of the input.
        channels: number of features/channels when the normalization layer requires this parameter
            but it is not specified in the norm parameters.
    """
    if name == "":
        return torch.nn.Identity()
    norm_name, norm_args = split_args(name)
    norm_type = Norm[norm_name, spatial_dims]
    kw_args = dict(norm_args)
    if has_option(norm_type, "num_features") and "num_features" not in kw_args:
        kw_args["num_features"] = channels
    if has_option(norm_type, "num_channels") and "num_channels" not in kw_args:
        kw_args["num_channels"] = channels
    return norm_type(**kw_args)



def has_option(obj: Callable, keywords: str | Sequence[str]) -> bool:
    """
    Return a boolean indicating whether the given callable `obj` has the `keywords` in its signature.
    """
    if not callable(obj):
        return False
    sig = inspect.signature(obj)
    return all(key in sig.parameters for key in ensure_tuple(keywords))


def ensure_tuple(vals: Any, wrap_array: bool = False) -> tuple:
    """
    Returns a tuple of `vals`.

    Args:
        vals: input data to convert to a tuple.
        wrap_array: if `True`, treat the input numerical array (ndarray/tensor) as one item of the tuple.
            if `False`, try to convert the array with `tuple(vals)`, default to `False`.

    """
    if wrap_array and isinstance(vals, (np.ndarray, torch.Tensor)):
        return (vals,)
    return tuple(vals) if issequenceiterable(vals) else (vals,)