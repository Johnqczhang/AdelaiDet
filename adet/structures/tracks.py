import itertools
import torch
from typing import Any, Dict, List, Union


class TrackState:
    Tentative = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class Tracks:
    """
    This class represents a list of tracks for multi-object tracking in a video sequence.
    It stores the attributes of tracks (e.g., ids, features, mean, covariance) as "fields".
    All fields must have the same ``__len__`` which is the number of tracks.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          tracks.ids = Tensor(...)
          print(tracks.mean)  # a tensor of shape (N, D)
          print('covariance' in tracks)

    2. ``len(tracks)`` returns the number of tracks

    3. Indexing: ``tracks[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Tracks`.
       Typically, ``indices`` is an integer vector of indices,
       or a binary mask of length ``num_tracks``

       .. code-block:: python

          tracks_activated = tracks[tracks.state == 1]
          tracks_unconfirmed = tracks[tracks.tentative == False]
    """

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """ Initiate new tracks (tracklets).

        Args:
            kwargs: fields to add to this `Tracks`.
        """
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Tracks!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of tracks,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Tracks of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this tracks.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Tracks":
        """
        Returns:
            Tracks: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Tracks()
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Tracks":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Tracks` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Tracks index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Tracks()
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        if len(self._fields) == 0:
            return 0
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()

    @staticmethod
    def cat(tracks_list: List["Tracks"]) -> "Tracks":
        """
        Args:
            tracks_list (list[Tracks])

        Returns:
            Tracks
        """
        assert all(isinstance(t, Tracks) for t in tracks_list)
        assert len(tracks_list) > 0
        if len(tracks_list) == 1:
            return tracks_list[0]

        tracks = Tracks()
        for k in tracks_list[0].get_fields():
            values = [t.get(k) for t in tracks_list]
            if isinstance(values[0], torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(values[0], list):
                values = list(itertools.chain(*values))
            elif hasattr(type(values[0]), "cat"):
                values = type(values[0]).cat(values)
            else:
                raise ValueError(f"Unsupported type {type(values[0])} for concatenation")
            tracks.set(k, values)
        return tracks

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += f"num_tracks={len(self)}, "
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__
